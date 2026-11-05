// TogetherCare web tier: one origin serving the built React app AND proxying
// the API, so a group of testers needs exactly one URL.
//
//   browser --https--> cloudflared --> :8080 (this) --> :8010 (FastAPI)
//
// Three things this exists to do, none of which a plain static server can:
//
//   1. Same origin. The SPA fetches "/api/..." on the host it was loaded from,
//      so there is no CORS preflight and no backend URL baked into the bundle
//      that goes stale every time the quick tunnel restarts.
//   2. Keep API_KEY on the server. The Cloudflare-Containers path writes it
//      into dist/env-config.js, which is page source -- every visitor could
//      read it and hit the shared GPU directly. Here the browser never sees
//      it; this process attaches it on the way through.
//   3. Gate access behind a shared passcode, since the tunnel is reachable by
//      anyone who has the link, VPN or not.
//
// Node stdlib only, deliberately: no npm install, nothing to keep patched.
import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import crypto from "node:crypto";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------- CONFIG ----------------------
const PORT = Number(process.env.PORT || 8080);
const DIST_DIR = path.resolve(
  process.env.DIST_DIR || path.join(__dirname, "..", "frontend", "dist")
);
const BACKEND_HOST = process.env.BACKEND_HOST || "127.0.0.1";
const BACKEND_PORT = Number(process.env.BACKEND_PORT || 8010);
// The backend's own x-api-key. Read here, never sent to the browser.
const BACKEND_API_KEY = process.env.BACKEND_API_KEY || "";

// The passcode handed out to testers.
const ACCESS_CODE = process.env.ACCESS_CODE || "";
// Signs the access cookie. Changing it logs everyone out, which is the
// intended way to revoke access without rotating the passcode itself.
const SESSION_SECRET = process.env.SESSION_SECRET || "";
const SESSION_HOURS = Number(process.env.SESSION_HOURS || 12);

// "auto" marks the cookie Secure only when the request arrived over https,
// which cloudflared reports via x-forwarded-proto. Without this, a Secure
// cookie is silently dropped by the browser when testing over plain
// http://localhost:8080 and the gate looks broken for no visible reason.
const COOKIE_SECURE = (process.env.COOKIE_SECURE || "auto").toLowerCase();

// Brute-force budget for the passcode. The code is the only thing standing
// between the open internet and a shared university GPU, so a wrong-guess
// loop is worth slowing down.
const MAX_ATTEMPTS = Number(process.env.MAX_ATTEMPTS || 10);
const LOCKOUT_MINUTES = Number(process.env.LOCKOUT_MINUTES || 15);

const COOKIE_NAME = "tc_access";

for (const [name, value] of [
  ["ACCESS_CODE", ACCESS_CODE],
  ["SESSION_SECRET", SESSION_SECRET],
]) {
  if (!value) {
    console.error(
      `[web] refusing to start: ${name} is not set. Without it the gate is ` +
        `either open to everyone or impossible to pass. See web/.env.example.`
    );
    process.exit(1);
  }
}
if (!fs.existsSync(path.join(DIST_DIR, "index.html"))) {
  console.error(
    `[web] refusing to start: no index.html in ${DIST_DIR}. ` +
      `Build it first: cd frontend && npm ci && npm run build`
  );
  process.exit(1);
}
if (!BACKEND_API_KEY) {
  // Not fatal: the backend treats an empty API_KEY as "auth disabled", and
  // this tier's own gate still applies. Worth saying out loud, though.
  console.warn("[web] BACKEND_API_KEY is empty - proxying without x-api-key");
}

const GATE_HTML = fs.readFileSync(path.join(__dirname, "gate.html"), "utf8");

// ---------------------- COOKIE / AUTH ----------------------
function sign(value) {
  return crypto.createHmac("sha256", SESSION_SECRET).update(value).digest("hex");
}

function issueToken() {
  const expires = Date.now() + SESSION_HOURS * 3600 * 1000;
  const payload = `v1.${expires}`;
  return `${payload}.${sign(payload)}`;
}

function tokenValid(token) {
  if (!token) return false;
  const parts = token.split(".");
  if (parts.length !== 3 || parts[0] !== "v1") return false;
  const [, expires, mac] = parts;
  const expected = sign(`v1.${expires}`);
  // Same length by construction (both hex sha256), so timingSafeEqual is safe
  // to call without a length check leaking through a thrown exception.
  if (mac.length !== expected.length) return false;
  if (!crypto.timingSafeEqual(Buffer.from(mac), Buffer.from(expected))) return false;
  return Number(expires) > Date.now();
}

function codeMatches(supplied) {
  // Compare digests rather than raw strings: equal-length inputs for
  // timingSafeEqual, and no early-exit on the first differing character.
  const a = crypto.createHash("sha256").update(String(supplied)).digest();
  const b = crypto.createHash("sha256").update(ACCESS_CODE).digest();
  return crypto.timingSafeEqual(a, b);
}

function parseCookies(header) {
  const out = {};
  for (const part of (header || "").split(";")) {
    const i = part.indexOf("=");
    if (i > 0) out[part.slice(0, i).trim()] = decodeURIComponent(part.slice(i + 1).trim());
  }
  return out;
}

function isAuthed(req) {
  return tokenValid(parseCookies(req.headers.cookie)[COOKIE_NAME]);
}

function wantsSecureCookie(req) {
  if (COOKIE_SECURE === "1" || COOKIE_SECURE === "true") return true;
  if (COOKIE_SECURE === "0" || COOKIE_SECURE === "false") return false;
  return (req.headers["x-forwarded-proto"] || "").split(",")[0].trim() === "https";
}

// ---------------------- RATE LIMIT ----------------------
const attempts = new Map(); // ip -> { count, until }

function clientIp(req) {
  // cloudflared puts the real client in x-forwarded-for; everything reaching
  // this process otherwise comes from 127.0.0.1 and would share one bucket.
  const fwd = (req.headers["x-forwarded-for"] || "").split(",")[0].trim();
  return fwd || req.socket.remoteAddress || "unknown";
}

function lockedOut(ip) {
  const rec = attempts.get(ip);
  if (!rec) return 0;
  if (rec.until && rec.until > Date.now()) return Math.ceil((rec.until - Date.now()) / 60000);
  if (rec.until && rec.until <= Date.now()) attempts.delete(ip);
  return 0;
}

function noteFailure(ip) {
  const rec = attempts.get(ip) || { count: 0, until: 0 };
  rec.count += 1;
  if (rec.count >= MAX_ATTEMPTS) {
    rec.until = Date.now() + LOCKOUT_MINUTES * 60 * 1000;
    rec.count = 0;
  }
  attempts.set(ip, rec);
}

// ---------------------- STATIC ----------------------
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".map": "application/json; charset=utf-8",
};

function serveStatic(req, res, urlPath) {
  let filePath = path.join(DIST_DIR, urlPath);
  // path.join normalises "..", so compare the result against the root to
  // reject traversal attempts rather than trusting the raw URL.
  if (filePath !== DIST_DIR && !filePath.startsWith(DIST_DIR + path.sep)) {
    res.writeHead(403).end("Forbidden");
    return;
  }

  fs.stat(filePath, (err, stats) => {
    // SPA fallback: unknown paths serve index.html so a refresh on a
    // client-side route still boots the app.
    if (err || !stats.isFile()) filePath = path.join(DIST_DIR, "index.html");
    const ext = path.extname(filePath);
    const headers = { "Content-Type": MIME[ext] || "application/octet-stream" };
    // Hashed asset filenames change on every build, so they are safe to cache
    // hard; index.html and env-config.js must not be, or a rebuilt app keeps
    // serving the old bundle to anyone who visited before.
    headers["Cache-Control"] = filePath.includes(`${path.sep}assets${path.sep}`)
      ? "public, max-age=31536000, immutable"
      : "no-store";
    res.writeHead(200, headers);
    fs.createReadStream(filePath).pipe(res);
  });
}

// ---------------------- PROXY ----------------------
const HOP_BY_HOP = new Set([
  "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
  "te", "trailers", "transfer-encoding", "upgrade",
]);

function proxyToBackend(req, res, backendPath) {
  const headers = {};
  for (const [k, v] of Object.entries(req.headers)) {
    const key = k.toLowerCase();
    if (HOP_BY_HOP.has(key)) continue;
    // host must be the backend's, not the tunnel hostname.
    if (key === "host") continue;
    // Never forward the browser's cookie (it carries the gate token, which is
    // this tier's business) or any client-supplied key -- this process is the
    // sole authority on what x-api-key the backend sees.
    if (key === "cookie" || key === "x-api-key") continue;
    // The bundle is served pre-compressed by nothing here; asking the backend
    // for gzip would mean buffering to re-encode, which breaks SSE streaming.
    if (key === "accept-encoding") continue;
    headers[k] = v;
  }
  if (BACKEND_API_KEY) headers["x-api-key"] = BACKEND_API_KEY;

  const upstream = http.request(
    { host: BACKEND_HOST, port: BACKEND_PORT, path: backendPath, method: req.method, headers },
    (backendRes) => {
      const outHeaders = {};
      for (const [k, v] of Object.entries(backendRes.headers)) {
        if (!HOP_BY_HOP.has(k.toLowerCase())) outHeaders[k] = v;
      }
      res.writeHead(backendRes.statusCode || 502, outHeaders);
      // /generate_stream is server-sent events: tokens must reach the browser
      // as they are produced. flushHeaders + setNoDelay stop Node from sitting
      // on small writes, which would otherwise deliver the whole answer at
      // once and lose the streaming effect the UI is built around.
      res.flushHeaders?.();
      backendRes.socket?.setNoDelay?.(true);
      res.socket?.setNoDelay?.(true);
      backendRes.pipe(res);
    }
  );

  upstream.on("error", (e) => {
    console.error(`[web] backend ${backendPath} failed: ${e.message}`);
    if (!res.headersSent) {
      res.writeHead(502, { "Content-Type": "application/json; charset=utf-8" });
      res.end(JSON.stringify({ detail: `backend unreachable: ${e.message}` }));
    } else {
      res.end();
    }
  });

  req.pipe(upstream);
}

// ---------------------- ROUTES ----------------------
function readBody(req, limitBytes = 4096) {
  return new Promise((resolve, reject) => {
    let data = "";
    let size = 0;
    req.on("data", (c) => {
      size += c.length;
      if (size > limitBytes) {
        reject(new Error("body too large"));
        req.destroy();
        return;
      }
      data += c;
    });
    req.on("end", () => resolve(data));
    req.on("error", reject);
  });
}

function sendJson(res, status, obj, extraHeaders = {}) {
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Cache-Control": "no-store",
    ...extraHeaders,
  });
  res.end(JSON.stringify(obj));
}

const server = http.createServer(async (req, res) => {
  const urlPath = decodeURIComponent((req.url || "/").split("?")[0]);
  const query = (req.url || "").includes("?") ? req.url.slice(req.url.indexOf("?")) : "";

  // Liveness for scripts/services.sh. No secrets, no auth, no backend call --
  // it answers even while the backend is still loading.
  if (urlPath === "/togethercareAI/web") {
    return sendJson(res, 200, {
      ok: true,
      service: "togethercareAI-web",
      dist_dir: DIST_DIR,
      backend: `${BACKEND_HOST}:${BACKEND_PORT}`,
      backend_api_key_set: Boolean(BACKEND_API_KEY),
      gated: true,
    });
  }

  // --- the gate ---
  if (urlPath === "/api/auth" && req.method === "POST") {
    const ip = clientIp(req);
    const waitMin = lockedOut(ip);
    if (waitMin) {
      return sendJson(res, 429, { error: `Too many attempts. Try again in ${waitMin} min.` });
    }
    let body;
    try {
      body = JSON.parse((await readBody(req)) || "{}");
    } catch {
      return sendJson(res, 400, { error: "Malformed request." });
    }
    if (!codeMatches(body.code || "")) {
      noteFailure(ip);
      console.warn(`[web] failed access-code attempt from ${ip}`);
      return sendJson(res, 401, { error: "That code is not right." });
    }
    attempts.delete(ip);
    const cookie = [
      `${COOKIE_NAME}=${issueToken()}`,
      "Path=/",
      "HttpOnly",
      "SameSite=Lax",
      `Max-Age=${SESSION_HOURS * 3600}`,
      wantsSecureCookie(req) ? "Secure" : "",
    ].filter(Boolean).join("; ");
    console.log(`[web] access granted to ${ip}`);
    return sendJson(res, 200, { ok: true }, { "Set-Cookie": cookie });
  }

  if (urlPath === "/api/logout") {
    const cookie = `${COOKIE_NAME}=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0`;
    return sendJson(res, 200, { ok: true }, { "Set-Cookie": cookie });
  }

  const authed = isAuthed(req);

  // --- API: proxied, but only for a holder of a valid gate cookie ---
  if (urlPath.startsWith("/api/")) {
    if (!authed) {
      // JSON, not the gate page: this is reached by fetch(), and an expired
      // session mid-conversation should surface as a clean error the UI can
      // show rather than an HTML blob parsed as JSON.
      return sendJson(res, 401, { detail: "Session expired. Reload and re-enter the access code." });
    }
    return proxyToBackend(req, res, urlPath.slice("/api".length) + query);
  }

  // --- static: nothing at all is served before the code is entered, so the
  // app bundle itself stays behind the gate ---
  if (!authed) {
    res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
    return res.end(GATE_HTML);
  }

  return serveStatic(req, res, urlPath);
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(
    `[web] serving ${DIST_DIR} on 127.0.0.1:${PORT}, ` +
      `proxying /api -> ${BACKEND_HOST}:${BACKEND_PORT}, ` +
      `gate on, session ${SESSION_HOURS}h`
  );
});

# TogetherCare AI

An empathy-focused RAG chatbot: a FastAPI backend that calls a fine-tuned
Llama-3 model on a Hugging Face Inference Endpoint and retrieves supporting
sources from Pinecone, fronted by a React chat UI.

This replaces the old Streamlit UI with a React (Vite) single-page app
deployed via **Cloudflare Containers**, connected to the FastAPI backend
(still running on your existing server) through a **Cloudflare Tunnel**.

```
backend/     FastAPI app (unchanged logic from streamlit_app/backend.py, + CORS)
frontend/    React chat UI, Dockerized, deployed via Cloudflare Containers
cloudflared/ Example Cloudflare Tunnel config for exposing the backend
```

## Why a tunnel?

Cloudflare Containers run on Cloudflare's network, not on your machine. The
frontend can no longer reach the backend at a private/internal address (like
the `129.108.x.x` addresses referenced in the old Streamlit setup) the way a
server-side Streamlit process could. Cloudflare Tunnel bridges that: it runs
*on your backend server*, makes an outbound-only connection to Cloudflare,
and gives you a public hostname to point the frontend at — no inbound
firewall ports, no public IP required.

## 1. Backend

```bash
cd backend
cp .env.example .env   # fill in HF_TOKEN, PINECONE_API_KEY, etc.
pip install -r requirements.txt
export $(grep -v '^#' .env | xargs)   # or use python-dotenv / your usual method
python -m uvicorn backend:app --host 0.0.0.0 --port 8000
```

Verify: `curl http://localhost:8000/health`

**CORS**: since the frontend is now a browser-side SPA (not a server-side
Streamlit process), the backend needs `ALLOWED_ORIGINS` set to wherever the
frontend is actually served from. Leave it as `*` while testing locally;
tighten it once you have the frontend's real Cloudflare Containers /
`workers.dev` URL (see step 3).

### Expose it publicly with Cloudflare Tunnel

Quick tunnel (no domain/account config needed, good for testing — URL
changes every restart):

```bash
cloudflared tunnel --url http://localhost:8000
```

This prints a `https://<random-words>.trycloudflare.com` URL. Use that as
`BACKEND_URL` for the frontend below.

Stable named tunnel (once you've added a domain to Cloudflare): see
`cloudflared/config.example.yml` for the fixed-hostname setup.

## 2. Frontend — local dev

```bash
cd frontend
npm install
npm run dev
```

Edit `public/env-config.js` (or set `VITE_BACKEND_URL` in a `.env` file) to
point at your backend's tunnel URL, e.g.:

```js
window.__ENV = { BACKEND_URL: "https://your-tunnel.trycloudflare.com" };
```

## 3. Frontend — deploy to Cloudflare Containers

Cloudflare Containers requires a **Workers Paid plan** and Docker running
locally (Wrangler builds and pushes the image for you).

> The Containers API moved fast during its beta. `worker/index.ts` and
> `wrangler.toml` here follow the documented pattern as of early 2026 —
> double check against https://developers.cloudflare.com/containers/ before
> your first deploy in case anything's shifted.

```bash
cd frontend
npm install @cloudflare/containers   # not pinned in package.json - grabs latest
npx wrangler login
```

Edit `wrangler.toml`'s `[vars]` block (or use `wrangler secret put API_KEY`
for the secret one) to point `BACKEND_URL` at your tunnel hostname, then:

```bash
npm run cf:deploy
```

Wrangler prints your frontend's URL (a `*.workers.dev` address until you
attach a custom domain). Take that URL and set it as `ALLOWED_ORIGINS` on the
backend, then restart the backend so CORS allows it.

### Changing config later without rebuilding

`BACKEND_URL`, `API_KEY`, `AB_ENABLED`, `AB_EVERY_N` are injected into the
running container at startup (`docker/entrypoint.sh` writes them into
`env-config.js`) from the Worker's env vars / `wrangler.toml` `[vars]`. Edit
those and redeploy the Worker (`npm run cf:deploy`) — no Docker rebuild
needed, since the Worker config is what's changing, not the image.

## Known quirks carried over from the original app

- **`AB_ENABLED` has an inverted default.** Both here and in the original
  Streamlit/backend code, it's computed as `env.get("AB_ENABLED", "0") == "0"`
  — which evaluates to *true* by default, and only turns A/B mode *off* if
  you set the var to something other than `"0"`. That's almost certainly a
  bug (the name implies `"1"` = on), but since it's how the app has actually
  been running, this port preserves it exactly rather than silently changing
  behavior. Worth deciding whether to fix it.
- **`pinecone` isn't pinned** in `backend/requirements.txt` — it wasn't
  present in the conda `requirements.txt` you exported, so check
  `pip show pinecone` in your working environment and pin it to match. This
  matters because `retrieve_top_k()` calls `results.get("matches", [])`,
  which assumes a dict-like response; that only works with the client
  version currently running in production.

## What's intentionally NOT in this repo

Model weights (`dpo_ft_llama3_emilia/`), local FAISS stores and their source
data (`faiss_default/`, `faiss_website_store/`, `chunks_processing/`,
`vector_db_data/`), and runtime logs/feedback (`backend/logs/`,
`backend/comments/`, gitignored) were left out on purpose — see the earlier
conversation for why. The Pinecone ingestion script
(`upload_faiss_to_pinecone.py`) and other one-off scripts from the old
workspace aren't copied here either; add them under a `scripts/` folder if
you want them version-controlled too.

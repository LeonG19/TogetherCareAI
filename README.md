# TogetherCare AI

An empathy-focused RAG chatbot: a FastAPI backend that calls a fine-tuned
Llama-3 model on a Hugging Face Inference Endpoint and retrieves supporting
sources from Pinecone, fronted by a React chat UI.

This replaces the old Streamlit UI with a React (Vite) single-page app. Both
the app and the API are served **from this server**, on one origin, and
exposed to testers through a **Cloudflare Tunnel**.

```
backend/     FastAPI app (unchanged logic from streamlit_app/backend.py, + CORS)
web/         Node web tier: serves the built app AND proxies /api. THE public entrypoint.
frontend/    React chat UI (Vite). Built to frontend/dist/, served by web/.
cloudflared/ Example Cloudflare Tunnel config
```

## How it is served

```
Browser (anywhere - UTEP VPN or not)
      |  one https:// URL + a shared access code
      v
cloudflared           outbound-only connection from this host
      v
127.0.0.1:8080        web/server.js
      |-- /           built React app (frontend/dist)
      +-- /api/*      reverse proxy, attaches x-api-key server-side
                v
        127.0.0.1:8010   FastAPI backend
                v
        127.0.0.1:8001   vLLM      +  Pinecone (hosted)
```

Three properties this buys, all of which matter for handing the link to a
group:

- **One origin.** The SPA fetches `/api/...` on whatever host it was loaded
  from. No CORS preflight, and no backend URL baked into the bundle that goes
  stale every time the quick tunnel restarts.
- **The API key stays on the server.** See the warning under
  [Access control](#access-control) — this is the part that is easy to get
  wrong.
- **No inbound ports.** cloudflared dials *out* to Cloudflare, so testers
  reach the app whether or not they are on UTEP's VPN, and the university
  firewall never needs an exception.

## Why a tunnel?

This host has no public IP and no inbound firewall exception, and half the
intended testers are off the university VPN. Cloudflare Tunnel bridges that:
`cloudflared` runs *on this server*, makes an outbound-only connection to
Cloudflare, and gets back a public hostname — no inbound ports, no public IP,
and nothing for a tester to install or join.

## 1. Backend

```bash
cd backend
cp .env.example .env   # fill in HF_ENDPOINT, HF_MODEL, HF_TOKEN
python -m venv .venv-backend && . .venv-backend/bin/activate
pip install -r requirements.txt
python -m uvicorn backend:app --host 127.0.0.1 --port 8000
```

`backend.py` loads `backend/.env` itself at import via `python-dotenv` with
`override=False`, so you no longer need the `export $(grep ... | xargs)` dance
— and a real environment variable still beats the file when both are set.

Verify:

```bash
curl http://localhost:8010/togethercareAI       # config echo: mode, endpoint, model
curl http://localhost:8010/togethercareAI/llm   # does one small REAL generation
```

> **Renamed from `/health`** at Dr. Pal's request. `/health` and `/health/llm`
> still answer as hidden aliases so that anything already pointed at them (the
> Cloudflare tunnel, an uptime monitor, an old bookmark) does not start 404ing
> — but `/togethercareAI` is the name to use, and the only one in the OpenAPI
> schema at `/docs`.

`/togethercareAI` never returns a secret — only `hf_token_set: true|false` and
`pinecone_api_key_set: true|false`. `/togethercareAI/llm` is the one to use
when something is wrong: a wrong `HF_MODEL` comes back as a 404 in the detail
string, a bad token as a 401, and an endpoint that is still loading as
"unreachable after N attempts". That distinguishes the three without opening
the UI.

### LLM-only vs. RAG mode

`RAG_ENABLED` (default `0`) decides whether retrieval exists at all.

| | `RAG_ENABLED=0` (default) | `RAG_ENABLED=1` |
|---|---|---|
| Install | `requirements.txt` | `requirements.txt` + `requirements-rag.txt` |
| Imports torch? | **No** | Yes |
| Needs a Pinecone key? | No | Only when `RAG_BACKEND=pinecone` |
| `retrieve_top_k()` | returns `[]` immediately | queries Pinecone or FAISS |
| Answers get a Sources section? | No | Yes |

`torch`, `sentence-transformers`, `numpy` and `pinecone` are imported inside
`_init_rag()`, not at module scope, so in the default mode they are never
imported and need not be installed. The slim install is ~20 packages and no
CUDA wheels. If you flip `RAG_ENABLED=1` without installing
`requirements-rag.txt`, you get an `ImportError` from `_init_rag()` at the
first request — that is deliberate, not a silent fallback to no sources.

### Retrieval (RAG) with a local FAISS index

`RAG_BACKEND` selects the vector store:

```
RAG_ENABLED=1
RAG_BACKEND=faiss
FAISS_INDEX_DIR=/mnt/lagarza3/llm-healthcare/indexes/faiss_default
RAG_MIN_SCORE=0.65
```

The index that ships with the research repo holds **165,430 chunks**
(`dim=384`, inner product over L2-normalised vectors, i.e. cosine) built from
`data/vector_db_data/` — alz.org and nia.nih.gov pages plus YouTube caregiver
transcripts. `embed_model.txt` records the encoder as
`sentence-transformers/all-MiniLM-L6-v2`, which is exactly `EMBED_MODEL_NAME`
in `backend.py`. `_init_rag()` **checks that stamp and refuses to start on a
mismatch**, because querying an index with a different encoder returns
plausible-looking nonsense rather than an error.

Cost at runtime: ~6 s startup and ~1.0 GB RSS to hold the index and metadata,
then ~8 ms per query. The encoder is pinned to `device="cpu"` on purpose — the
GPU is shared and vLLM has claimed nearly all free VRAM. Install CPU-only
torch (see `requirements-rag.txt`); the CUDA build costs ~4 GB for nothing.

### Retrieval with hosted Pinecone

`RAG_BACKEND=pinecone` is the current deployment setting:

```
RAG_ENABLED=1
RAG_BACKEND=pinecone
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=empathy-rag
PINECONE_NAMESPACE=          # empty = the index's default namespace
RAG_MIN_SCORE=0.65
```

The `empathy-rag` index holds **173,793 vectors**, `dimension=384`,
`metric=cosine` — the same encoder (`all-MiniLM-L6-v2`) and the same corpus
family as the local FAISS index, so `RAG_MIN_SCORE` carries over unchanged.
Metadata on each vector is flat: `title`, `url`, `text`, `source_id`, which is
exactly the shape `extract_url()` and `format_sources()` already read.

**Startup validation.** `_init_rag()` makes one `describe_index_stats()` call
before serving. It fails loudly if `PINECONE_API_KEY` is unset, if the key or
`PINECONE_INDEX_NAME` is wrong, or if the index dimension is not 384. This
matters more than it looks: `retrieve_top_k()` is wrapped in `try/except` at
all four call sites, so a store that only fails on first use produces a
chatbot that answers normally and silently *never cites anything* — the exact
failure mode hardest to notice from the UI. Failing at boot converts that into
a unit that will not start.

The FAISS branch gets the same protection from the `embed_model.txt` stamp.
Pinecone carries no encoder stamp, so dimension is the only automatic check
available — it catches a wholly wrong encoder, but not a *different* 384-dim
model. If you rebuild the index, rebuild it with `EMBED_MODEL_NAME`.

**Pinecone vs. FAISS.** FAISS `IndexFlat` is *exact* search and therefore the
recall ceiling, with no key, no network hop and no quota; Pinecone adds a
network round trip per query but needs no 396 MB index directory shipped with
the deployment, which is what makes it the portable choice. Both branches are
live and switching is a one-line `.env` change — `FAISS_INDEX_DIR` is still
set, so `RAG_BACKEND=faiss` reverts without touching anything else.

Client note: `pinecone` v10 returns a `QueryResponse` object, not a dict —
`results.get("matches", [])` silently yields nothing on it. `_extract_matches()`
accepts both shapes deliberately.

#### What this retrieval actually does

Read this before assuming answers are grounded. `retrieve_top_k()` is called
with the **generated answer**, *after* generation, at all four call sites, and
its output is only used to append a Sources block. **The model never sees the
retrieved text.** This is post-hoc source attribution, not
retrieval-augmented generation — answers come from the fine-tuned model's own
knowledge and citations are attached afterwards. Preserved as-is because it is
what the original app did; converting it to true context injection is a real
behavioural change and has not been made.

#### RAG_MIN_SCORE

Only chunks scoring at or above this cosine threshold are cited. Measured over
realistic answers on this corpus: median top-5 score ~0.74, range 0.53–0.86.
At `0.70` about half of retrieved chunks survive; `0.65` keeps most genuinely
relevant ones while still returning **no** sources for queries the corpus does
not cover (e.g. "a parent who repeats questions" tops out at 0.56). An empty
Sources block is therefore normal and correct, not a failure.

#### HF_ENDPOINT vs HF_HUB_ENDPOINT — a genuine trap

`huggingface_hub` reads the environment variable **`HF_ENDPOINT`** as the Hub
base URL. This app has used that same name since before RAG existed, to mean
the *LLM inference endpoint*. With RAG off the collision is invisible, because
`huggingface_hub` is never imported. Turn RAG on while `HF_ENDPOINT` points at
a local vLLM server and sentence-transformers tries to download the encoder
**from the vLLM server**, failing with a 404 on
`{HF_ENDPOINT}/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json`.

`_init_rag()` shadows `HF_ENDPOINT` with `HF_HUB_ENDPOINT` for the duration of
the encoder load and restores it afterwards. Do not "simplify" that block.

### Pointing the backend at a local vLLM server

Any OpenAI-compatible `/v1/chat/completions` server works, so the same
backend talks to either the hosted HF Inference Endpoint or a local vLLM
instance — only `.env` changes:

```
HF_ENDPOINT=http://127.0.0.1:8001
HF_MODEL=togethercare-llama3-8b
HF_TOKEN=<the key vLLM was started with>
RAG_ENABLED=0
```

See `scripts/run_vllm.sh` to serve the model and `scripts/smoke_test.sh` to
verify the whole path. `scripts/merge_lora.py` merges a PEFT/LoRA adapter into
the base model first — vLLM is pointed at the *merged* directory, not at an
adapter directory.

Both the HF endpoint (which answers `503` while waking from scale-to-zero) and
a local vLLM server (which refuses the connection outright for the several
minutes it spends loading an 8B checkpoint) are retried by `_hf_post()`,
controlled by `HF_MAX_RETRIES` / `HF_RETRY_WAIT` (default ~2 minutes total).
Each retry is logged. After the budget is exhausted the request fails with a
502 that names the endpoint and model.

**CORS**: `ALLOWED_ORIGINS` names the origins allowed to call the backend
from a browser. In the deployed setup this barely matters: the browser only
ever talks to the web tier on its own origin, and the proxy's request to the
backend is server-to-server, so no CORS check happens at all. It matters for
`npm run dev`, which *is* cross-origin — hence `http://localhost:5173`.

### Expose it publicly with Cloudflare Tunnel

Quick tunnel (no domain/account config needed, good for testing — URL
changes every restart):

```bash
cloudflared tunnel --url http://localhost:8080   # 8080 = the web tier, NOT the backend
```

This prints a `https://<random-words>.trycloudflare.com` URL. That single URL
serves the app *and* its API — hand it to testers along with the access code.
In normal use `togethercare-tunnel.service` does this for you.

Stable named tunnel — a **fixed** hostname that survives restarts. Needs a
domain already added to your Cloudflare account. One interactive step, then
one command:

```bash
/home/lgarza3/bin/cloudflared tunnel login        # opens a browser link; pick your zone
./scripts/setup_named_tunnel.sh togethercareai.yourdomain.com
```

That creates (or reuses) a tunnel named `togethercare`, points the DNS record
at it, writes `~/.cloudflared/config.yml`, and swaps
`togethercare-tunnel.service` over to it — publishing only port 8080, same as
before. It backs up the quick-tunnel unit first, so
`./scripts/setup_named_tunnel.sh --revert` puts things back exactly as they
were. `cloudflared/config.example.yml` documents the same setup by hand.

## 1b. Running it detached (systemd --user)

Four `systemctl --user` units run the stack. Lingering is enabled for the
account (`loginctl enable-linger`, which went through polkit without sudo on
this box), so **they survive logout and reboot** and are not tied to the SSH
session that started them.

| unit | what | port |
|---|---|---|
| `togethercare-vllm` | vLLM serving the merged model | 127.0.0.1:8001 |
| `togethercare-backend` | FastAPI backend | 127.0.0.1:8010 |
| `togethercare-web` | app + `/api` proxy + access gate | 127.0.0.1:8080 |
| `togethercare-tunnel` | cloudflared quick tunnel → **8080** | public |

Unit files are version-controlled in `systemd/`; copy them to
`~/.config/systemd/user/` and run `./scripts/services.sh install` on a fresh
clone.

```bash
./scripts/services.sh start      # start all four
./scripts/services.sh status     # units + port health + current tunnel URL
./scripts/services.sh share      # the URL AND the access code, ready to send
./scripts/services.sh url        # just the public URL
./scripts/services.sh logs       # tail all four logs
./scripts/services.sh stop
```

Logs are plain files under `/mnt/lagarza3/logs/{vllm,backend,web,tunnel}.log`.

Two details that matter:

- `Restart=always` on all four, and `TimeoutStartSec=900` on vLLM because an
  8B checkpoint takes ~2 minutes to load and vLLM does not bind its port until
  it is finished. Without that, systemd would call a normal start a failure.
- The backend uses `Wants=`, not `Requires=`, on vLLM. It must come up and stay
  up while vLLM is still loading — `_hf_post()` retries connection-refused for
  ~2 minutes and `/togethercareAI` stays answerable throughout. The web tier
  uses `Wants=` on the backend for the same reason: the gate and the app must
  still load while the backend restarts, so a tester sees a 502 on one message
  rather than a dead site.

### Ports on this host

| port | what | exposed? |
|---|---|---|
| `8001` | vLLM | **never** — loopback only |
| `8010` | FastAPI backend | no — only the web tier talks to it |
| `8080` | web tier | **yes, this is the tunnelled one** |
| `8000`, `8002` | **another user's services** | do not touch, do not tunnel |

`8000` and `8002` belong to other users on this shared box — the backend was
moved off the conventional 8000 for exactly that reason. Do not tunnel 8000;
you would be publishing someone else's service.

Note that 8010 is no longer tunnelled. Exposing the backend directly would let
anyone with the hostname skip the access gate entirely.

### Public access via cloudflared

`scripts/services.sh url` prints the current public hostname. By default that
is a `https://<random-words>.trycloudflare.com` quick tunnel, which **gets a
new random hostname every restart** — it is not stable. For a fixed hostname
you need a named tunnel against a Cloudflare-hosted domain: run
`./scripts/setup_named_tunnel.sh <hostname>` (see §1a). Once that is in place
`services.sh url` reads the hostname straight out of `~/.cloudflared/config.yml`
instead of scraping the log.

Only the web tier is exposed. vLLM stays on `127.0.0.1:8001`, and the backend
on `127.0.0.1:8010`, neither reachable from the campus network.

**`API_KEY` must still be set** in `backend/.env`. With it empty,
`auth_check()` is a no-op — and while the web tier's gate would still be in
front, anything that ever reaches 8010 directly (a future tunnel, another
process on this shared host) would have unauthenticated access to the GPU.
`/generate`, `/generate_stream`, `/generate_ab` and `/togethercareAI/llm` all
require `x-api-key`; `web/server.js` attaches it. `/togethercareAI` is
deliberately open for liveness checks — no inference, no secrets.

## 2. Frontend — build it for the server

There is no system Node on this host and no root to add one, so Node 20 was
unpacked from the official tarball into `/mnt/lagarza3/node` (no root needed):

```bash
mkdir -p /mnt/lagarza3/node
curl -fsSL https://nodejs.org/dist/v20.18.1/node-v20.18.1-linux-x64.tar.xz \
  | tar -xJ -C /mnt/lagarza3/node --strip-components=1
```

Then build:

```bash
./scripts/build_frontend.sh
```

That runs `npm ci` (first time only) and `npm run build`, then **overwrites
`dist/env-config.js`** with the values this deployment needs:

```js
window.__ENV = { BACKEND_URL: "/api", API_KEY: "", ... };
```

`BACKEND_URL: "/api"` is what makes the single-origin setup work — the app
calls back to whatever host served it, so a new quick-tunnel hostname needs no
rebuild and no config edit. `API_KEY` stays **empty on purpose**; see
[Access control](#access-control).

`web/server.js` reads `dist/` from disk per request, so a rebuild takes effect
without restarting anything.

### Local dev

```bash
export PATH=/mnt/lagarza3/node/bin:$PATH
cd frontend && npm install && npm run dev
```

`npm run dev` serves on :5173 and uses the checked-in `public/env-config.js`,
which still points at `http://localhost:8000` — set `BACKEND_URL` there (and
`API_KEY`, in dev only) to hit a backend directly, bypassing the web tier.
Note the backend's `ALLOWED_ORIGINS` must include `http://localhost:5173` for
that path, because dev *is* cross-origin; the deployed path is not.

## 3. The web tier

`web/server.js` is Node stdlib only — no dependencies, nothing to `npm
install`, nothing to keep patched. It does three things:

| | |
|---|---|
| `GET /togethercareAI/web` | liveness. No auth, no secrets, no backend call — answers even while the backend is reloading. |
| `POST /api/auth` | the gate. Correct code → a signed, `HttpOnly` cookie. |
| `/api/*` | reverse proxy to `127.0.0.1:8010`, **minus** the `/api` prefix, **plus** `x-api-key`. |
| everything else | `frontend/dist/`, with SPA fallback — but only once the cookie exists. |

Config is `web/.env` (copy `web/.env.example`). The server refuses to start
if `ACCESS_CODE` or `SESSION_SECRET` is missing, rather than coming up with an
open or unusable gate.

### Access control

> **The trap this replaces.** The Cloudflare-Containers path writes `API_KEY`
> into `dist/env-config.js` via `docker/entrypoint.sh`. That file is page
> source. Every visitor could read the key protecting a shared university GPU
> and then call the backend directly, without the UI, forever. Serving the app
> from a tier that holds the key server-side is the whole reason `web/` exists.

Testers get **one URL and one shared code**. On success the server sets a
signed `HttpOnly` cookie (`SESSION_HOURS`, default 12) — the code itself is
never stored in the browser, and JavaScript cannot read the cookie.

Until that cookie exists, **nothing** is served: not the app shell, not the JS
bundle. `/api/*` answers `401` as JSON rather than HTML, so a session that
expires mid-conversation surfaces as an error the UI can show instead of an
HTML blob parsed as JSON.

Wrong guesses are rate-limited per client IP (`MAX_ATTEMPTS`, default 10, then
a `LOCKOUT_MINUTES` freeze), taken from `x-forwarded-for` — without that,
every request would arrive from `127.0.0.1` via cloudflared and share one
bucket. Both the code comparison and the cookie signature use
`crypto.timingSafeEqual`.

To rotate: change `ACCESS_CODE` in `web/.env` and restart. To revoke every
existing session without changing the code, change `SESSION_SECRET` instead —
that invalidates all outstanding cookies immediately.

### Handing the link out

```bash
./scripts/services.sh share
#   URL:  https://<random-words>.trycloudflare.com
#   Code: togethercare-xxxxxx
```

**A quick tunnel gets a new random hostname on every restart**, so re-run
`share` and re-send the link after any reboot or `services.sh restart`. To
stop that happening — before a demo, say — switch to a named tunnel with
`./scripts/setup_named_tunnel.sh togethercareai.yourdomain.com`; the hostname
is then fixed and `share` keeps printing it unchanged.

### Note on the Cloudflare Containers path

`frontend/Dockerfile`, `worker/index.ts` and `wrangler.toml` are still in the
tree but are **superseded by `web/`** for this deployment. They require a
Workers Paid plan, and they carry the `API_KEY`-in-page-source problem
described above. Left in place rather than deleted, since nothing depends on
them and removing them is a separate decision.

## Known quirks carried over from the original app

- **`AB_ENABLED` has an inverted default, in *both* the backend and the
  frontend.** It is computed as `== "0"` in each of:
  - `backend/backend.py` — `AB_ENABLED = os.environ.get("AB_ENABLED", "0") == "0"`
  - `frontend/src/config.ts` — `getAbEnabled()`, `return v === "0"`

  So **A/B mode is ON unless the variable is set to something other than
  `"0"`**. Unset means on; `AB_ENABLED=0` means on; `AB_ENABLED=1` means
  *off*, which is the opposite of what the name implies. That is almost
  certainly a bug, but it is how the app has actually been running, so this
  port preserves it exactly rather than silently changing behavior. **Not
  fixed in the local-vLLM pass** — deliberately left alone so that change
  lands on its own, and note that fixing it requires changing both files
  together or the two halves will disagree about what mode they are in.
- **`pinecone` is now pinned** to `10.0.0` in `backend/requirements-rag.txt`,
  which is the version installed in `.venv-backend-rag` and the one the
  Pinecone path was verified against.
  The response-shape half of this is fixed: `retrieve_top_k()` used to call
  `results.get("matches", [])`, which only works on a pre-v3 client and
  silently returns *nothing* on a current one (v3+ returns a `QueryResponse`
  object with attribute access and `ScoredVector` entries, not a dict).
  `_extract_matches()` now accepts both shapes — which is what `RAG_BACKEND=
  pinecone` depends on, since v10 is exactly the object-returning client.

## What's intentionally NOT in this repo

Model weights (`dpo_ft_llama3_emilia/`), local FAISS stores and their source
data (`faiss_default/`, `faiss_website_store/`, `chunks_processing/`,
`vector_db_data/`), and runtime logs/feedback (`backend/logs/`,
`backend/comments/`, gitignored) were left out on purpose — see the earlier
conversation for why. The Pinecone ingestion script
(`upload_faiss_to_pinecone.py`) and other one-off scripts from the old
workspace aren't copied here either; add them under a `scripts/` folder if
you want them version-controlled too.

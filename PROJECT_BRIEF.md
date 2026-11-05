# TogetherCare AI — Project Brief

A single-file knowledge base for this project. Written to be uploaded into a
Claude Project so an assistant can answer questions about what this is, how it
was built, how it is served, and where the bodies are buried.

**Contains no secrets.** API keys, the tester access code, and HF tokens are
referenced by variable name only.

---

## 1. What it is

An empathy-focused question-answering chatbot for **caregivers, nurses and
doctors dealing with geriatrics, dementia and Alzheimer's disease**.

The thesis of the project: for this audience, *how* an answer is delivered
matters as much as what it says. A clinically correct but cold answer is a bad
answer. So the work centers on fine-tuning a model to open with a brief,
compassionate acknowledgement and then explain simply, and on measuring
empathy and reading level as first-class metrics alongside accuracy.

Two repositories, two jobs:

| Repo | Path | Role |
|---|---|---|
| `llm-healthcare` | `/mnt/lagarza3/llm-healthcare` | **Research.** Training, datasets, evaluation harness, results. Public on GitHub (`LeonG19/llm-healthcare`). |
| `togethercare` | `/mnt/lagarza3/togethercare` | **Product.** The deployed app: backend, web tier, React frontend, systemd units, ops scripts. |

Stakeholder note: a reviewer referred to as **Dr. Pal** asked for the health
endpoint to be renamed from `/health` to `/togethercareAI`. That rename is why
the odd endpoint name exists; `/health` still answers as a hidden alias.

---

## 2. Current status (as of 2026-09-09)

**Live and demo-ready** at **https://togethercareai.com**, behind a shared
access code. Verified end-to-end the morning of 2026-09-09: 20/20 smoke-test
checks passing locally and through the public URL.

The public hostname is a **named** Cloudflare tunnel, so it is stable across
restarts. (An earlier quick tunnel produced a new random
`*.trycloudflare.com` hostname on every restart — that was the motivation for
switching before demos.)

---

## 3. Architecture

```
Browser (anywhere — no VPN needed)
      |  one https:// URL + one shared access code
      v
Cloudflare  <-- cloudflared dials OUT from the host; no inbound ports, no public IP
      v
127.0.0.1:8080   web/server.js   (Node stdlib only, zero dependencies)
      |-- /          built React app from frontend/dist
      +-- /api/*     reverse proxy -> strips /api, ATTACHES x-api-key server-side
                v
        127.0.0.1:8010   FastAPI backend (backend/backend.py)
                v
        127.0.0.1:8001   vLLM (merged fine-tuned model)  +  Pinecone (hosted)
```

Three properties this design buys:

1. **One origin.** The SPA fetches `/api/...` relative to whatever host served
   it, so no CORS preflight and no backend URL baked into the bundle.
2. **The API key never reaches the browser.** The web tier holds it and
   attaches it server-side. This is the whole reason the `web/` tier exists.
3. **No inbound ports.** Works for testers on or off the university VPN, with
   no firewall exception.

### Ports on this shared host

| Port | What | Exposed? |
|---|---|---|
| 8001 | vLLM | Never — loopback only |
| 8010 | FastAPI backend | No — only the web tier talks to it |
| 8080 | Web tier | **Yes — this is the tunnelled one** |
| 8000, 8002 | **Another user's services** | Do not touch, do not tunnel |

The backend was deliberately moved off the conventional port 8000 because
another user on this shared box already occupies it. Tunnelling 8000 would
publish someone else's service.

---

## 4. The model

**Served model:** `togethercare-llama3-8b` — Llama-3.1-8B-Instruct fine-tuned
with **DPO** (Direct Preference Optimization), LoRA adapter **merged into the
base weights** and served by vLLM from
`/mnt/lagarza3/models/togethercare-llama3-8b` (~16 GB, bfloat16).

vLLM is pointed at the *merged* directory, not at an adapter directory —
`scripts/merge_lora.py` does the merge.

### Training method

QLoRA + DPO, via `train/dpo_LLM.py` (TRL `DPOTrainer`):

| Setting | Value |
|---|---|
| Base | `meta-llama/Llama-3.1-8B-Instruct` |
| Quantization | 4-bit NF4, double-quant, bfloat16 compute |
| LoRA | r=32, alpha=64, dropout=0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention only, no MLP) |
| DPO beta | 0.1 |
| Learning rate | 2e-4 |
| Batch / grad accum | 2 / 16 (effective 32) |
| Max length | 2048 |
| Epochs | 3 completed (375 steps at the shipped checkpoint) |

**Training system prompt** (also saved as `DPO_PROMPT.json` next to the
weights, so it travels with the model):

> "You are a careful, evidence-based clinical assistant specialized in
> healthcare, geriatrics, dementia, and Alzheimer's disease. Start your answer
> with a brief, compassionate one-line sentence acknowledging the user's
> concern, then provide a clear, simple, and friendly explanation. Do not
> invent facts. Be polite and kind."

### Four base models were compared

DPO was run against **Llama-3.1-8B, Mistral, DeepSeek and Qwen** (weights in
`train/dpo_ft_*_weights/`). Llama-3.1-8B won and is what ships. There is also
an SFT track (`train/sft_LLM.py`, `sft_ft_llama3_weights/`) that predates the
DPO work.

### Honest caveat about the training signal

The training logs show `rewards/accuracies` hitting **1.0** and loss falling
to **0.0** by roughly epoch 2, with reward margins around 30–50 from the very
first logged steps. That is not a sign of an unusually good model — it means
the preference pairs are **trivially separable**. Looking at the data, the
`chosen` responses are warm and supportive while the `rejected` ones are
openly condescending ("How quaint."). The model learned a *style* distinction
that was easy to learn.

The practical reading: DPO here reliably bought **tone, warmth and reading
level**. It should not be claimed to have improved clinical correctness, and
the evaluation numbers below are consistent with that.

---

## 5. Data

`llm-healthcare/data/`:

**QA / fine-tuning** (`data/QA_healthcare/`) — assembled from public medical QA
sets plus generated preference pairs:

- `filtered_qa_pairs_MedQuAD.csv`, `MedQuAD2` — MedQuAD
- `filtered_qa_pairs_MedMCQA.csv`, `MedMCQA2` — MedMCQA
- `filtered_Alzheimer_Dementia_QAs.csv`, `alzheimers_qa_dataset.csv` — domain-specific
- `qa_with_llm_final.jsonl` (503 rows), `qa_with_llm_final2.jsonl` (999 rows) —
  the DPO preference data, shaped `{prompt, chosen, rejected}`
- `500_QA.csv` / `500_QA_chosen.csv` (504 rows)
- `test.csv` (524 rows) — the held-out evaluation set

**Retrieval corpus** (`data/vector_db_data/`):

- `combined_website_rag.jsonl` — alz.org and nia.nih.gov pages
- `combined_youtube_rag_part1..5.jsonl` — YouTube caregiver transcripts

---

## 6. Retrieval (RAG) — read this carefully

### The critical caveat

**This is post-hoc source attribution, not retrieval-augmented generation.**

`retrieve_top_k()` is called with the **generated answer**, *after* generation,
at all four call sites, and its output is used only to append a Sources block.
**The model never sees the retrieved text.** Answers come from the fine-tuned
model's own parametric knowledge; citations are attached afterwards.

This was preserved deliberately because it is what the original app did.
Converting it to true context injection is a real behavioural change and has
not been made. **Do not describe this system as grounded generation.** If
asked in a demo whether answers are grounded in the cited sources, the honest
answer is no — the citations are related material found after the fact.

### Configuration

Two interchangeable backends, switched by a one-line `.env` change:

| | Pinecone (**current**) | FAISS |
|---|---|---|
| Index | `empathy-rag`, **173,793 vectors** | `faiss_default`, **165,430 chunks** |
| Dim / metric | 384 / cosine | 384 / inner product on L2-normalised vectors |
| Search | Approximate, hosted | `IndexFlat` — **exact**, the recall ceiling |
| Cost | Network round trip per query | ~6 s startup, ~1.0 GB RSS, ~8 ms/query |
| Why | No 396 MB index dir to ship — portable | No key, no network, no quota |

Encoder for both: **`sentence-transformers/all-MiniLM-L6-v2`**, pinned to
`device="cpu"` on purpose because the GPU is shared and vLLM has claimed the
VRAM. Install CPU-only torch; the CUDA build costs ~4 GB for nothing.

`RAG_MIN_SCORE = 0.65`. Measured over realistic answers on this corpus: median
top-5 score ~0.74, range 0.53–0.86. **An empty Sources block is normal and
correct**, not a failure — a query the corpus does not cover (e.g. "a parent
who repeats questions") tops out around 0.56 and is correctly filtered out.

### Two guardrails worth knowing

- **Encoder-mismatch stamp.** The FAISS index records its encoder in
  `embed_model.txt` and `_init_rag()` **refuses to start on a mismatch** —
  querying an index with a different encoder returns plausible-looking
  nonsense rather than an error. Pinecone carries no such stamp, so dimension
  is the only automatic check there.
- **Boot-time validation.** `_init_rag()` makes one `describe_index_stats()`
  call before serving. This matters because `retrieve_top_k()` is wrapped in
  try/except at all four call sites — a store that only fails on first use
  yields a chatbot that answers normally and *silently never cites anything*,
  the failure mode hardest to spot from the UI. Failing at boot converts that
  into a unit that will not start.

---

## 7. Evaluation

Harness: `source_code/main.py` + `config.py`. Test set: `test.csv` (524 rows).
Generation at temperature 0.3, max 150 tokens. Results in `results/`.

Metrics: ROUGE-L, BLEU, BERTScore (rescaled with baseline), sentence-transformer
semantic similarity (`st_sem`), embedding cosine, **GEval** and **g_empathy**
(LLM-as-judge via DeepEval), an **NLI** entailment score
(`MoritzLaurer/deberta-v3-base-zeroshot-v2.0`), and **FKGL** reading grade
level via `textstat`.

### Base vs. DPO fine-tuned, per base model

| Metric | Llama base | **Llama DPO** | Mistral base | Mistral DPO | DeepSeek base | DeepSeek DPO |
|---|---|---|---|---|---|---|
| ROUGE-L | 0.225 | **0.250** | 0.199 | 0.218 | 0.145 | 0.174 |
| BLEU | 11.84 | **16.13** | 6.18 | 7.69 | 3.15 | 4.08 |
| BERTScore F1 | 0.250 | **0.309** | 0.228 | 0.263 | 0.031 | 0.155 |
| ST semantic | 0.799 | **0.839** | 0.770 | 0.847 | 0.698 | 0.714 |
| Embed cosine | 0.794 | **0.820** | 0.778 | 0.818 | 0.665 | 0.732 |
| GEval | 0.731 | **0.746** | 0.707 | 0.780 | 0.523 | 0.743 |
| NLI | — | 0.844 | 0.793 | 0.850 | 0.783 | 0.821 |
| Empathy (g_empathy) | 0.697 | **0.724** | 0.707 | 0.781 | 0.523 | 0.743 |

**DPO improved every metric on every base model.** Llama-3.1-8B is the
strongest across the board, which is why it ships.

### Reading level (FKGL — lower is more accessible)

| | Base | DPO |
|---|---|---|
| Llama | 12.65 | **11.98** |
| DeepSeek | 10.37 | 9.35 |
| Mistral | — | 9.35 |

DPO made answers easier to read in every case. For context, US public-health
guidance typically targets grade 8 or below, so even the fine-tuned Llama at
~12 remains **above** the ideal reading level for caregivers. That is a fair,
known limitation and a natural "what's next" item.

### RAG configuration run

`rag_base_test_k3_v1` (FAISS, k=3): ROUGE-L 0.224, BLEU 6.93, BERTScore 0.263,
ST semantic 0.827, embed cosine 0.809, **GEval 0.767** — the highest GEval of
any run. But note `rag_reading` FKGL is **13.64**, the worst reading level
measured: appending source material makes answers denser and harder to read.

### Data-quality caveats in the results (be honest about these)

- `base_deep_mod` reports `ModLLM` of **4.63** while every other ModLLM value
  sits in 0.72–0.85. That run is off-scale and should be treated as broken,
  not as a real score.
- `dpo_fkgl_mistral` and `dpo_fkgl_deep` are **byte-identical** (9.348590...),
  which almost certainly means one result was copied or a run was mislabelled.
  The Mistral reading-level number should be re-run before being quoted.
- Some base/DPO cells are missing (no Llama base NLI or ModLLM), so a few
  comparisons are one-sided.

---

## 8. Serving and operations

### Four `systemctl --user` units

| Unit | What | Port |
|---|---|---|
| `togethercare-vllm` | vLLM serving the merged model | 127.0.0.1:8001 |
| `togethercare-backend` | FastAPI backend | 127.0.0.1:8010 |
| `togethercare-web` | App + `/api` proxy + access gate | 127.0.0.1:8080 |
| `togethercare-tunnel` | cloudflared named tunnel → 8080 | public |

Lingering is enabled (`loginctl enable-linger`), so these **survive logout and
reboot** and are not tied to an SSH session.

Two deliberate details:

- `Restart=always` on all four, and `TimeoutStartSec=900` on vLLM because an
  8B checkpoint takes ~2 minutes to load and vLLM does not bind its port until
  it finishes. Without that, systemd would call a normal start a failure.
- The backend uses `Wants=`, not `Requires=`, on vLLM — it must come up and
  stay up while vLLM is still loading. `_hf_post()` retries connection-refused
  for ~2 minutes and `/togethercareAI` stays answerable throughout. The web
  tier uses `Wants=` on the backend for the same reason: a tester sees a 502
  on one message rather than a dead site.

### Commands

```bash
./scripts/services.sh start|stop|status|logs
./scripts/services.sh share    # public URL + access code, ready to send
./scripts/services.sh url      # just the URL
./scripts/smoke_test.sh        # 16-section end-to-end check, 20 assertions
WEB=https://togethercareai.com ./scripts/smoke_test.sh   # same, through Cloudflare
```

Logs are plain files in `/mnt/lagarza3/logs/{vllm,backend,web,tunnel}.log`.
Note: the vLLM log **contains the API key** in its startup banner.

### Health endpoints

| Endpoint | Auth | Does |
|---|---|---|
| `GET /togethercareAI` | none | Config echo: mode, endpoint, model, vector count. Never returns a secret — only `hf_token_set: true\|false`. Open on purpose for liveness checks. |
| `GET /togethercareAI/llm` | **`x-api-key`** | One small **real** generation. Gated because it burns GPU time. |
| `GET /togethercareAI/web` | none | Web-tier liveness; answers even while the backend reloads. |

`/togethercareAI/llm` is the diagnostic that distinguishes failures without
opening the UI: wrong `HF_MODEL` → 404 in the detail string, bad token → 401,
endpoint still loading → "unreachable after N attempts".

`/health` and `/health/llm` remain as hidden aliases so existing probes and
bookmarks do not start 404ing.

### API surface

`POST /generate`, `POST /generate_stream` (SSE), `POST /generate_ab`,
`POST /ab_vote`, `POST /comment`. All require `x-api-key`; the web tier
attaches it. `/generate_stream` is what the UI actually uses.

### Hardware

3× NVIDIA RTX 6000 Ada (49 GB each), shared with other users. vLLM runs on
GPU 0 with `--gpu-memory-utilization 0.42` (~21 GB) and `--max-model-len 8192`,
deliberately leaving room for other tenants.

---

## 9. Access control

Testers get **one URL and one shared code**.

- Correct code → a signed, `HttpOnly` cookie (`SESSION_HOURS`, default 12).
  The code itself is never stored in the browser and JavaScript cannot read
  the cookie.
- Until that cookie exists, **nothing** is served — not the app shell, not the
  JS bundle.
- `/api/*` answers `401` as **JSON**, not HTML, so an expired session surfaces
  as an error the UI can display instead of an HTML blob parsed as JSON.
- Wrong guesses are rate-limited per client IP (`MAX_ATTEMPTS`, default 10,
  then a `LOCKOUT_MINUTES` freeze), taken from `x-forwarded-for` — without
  that, every request arrives from `127.0.0.1` via cloudflared and shares one
  bucket.
- Both the code comparison and the cookie signature use
  `crypto.timingSafeEqual`.

**To rotate:** change `ACCESS_CODE` in `web/.env` and restart. **To revoke all
existing sessions** without changing the code, change `SESSION_SECRET`.

### The trap this design replaces

An earlier Cloudflare-Containers path wrote `API_KEY` into
`dist/env-config.js` via `docker/entrypoint.sh`. **That file is page source.**
Every visitor could read the key protecting a shared university GPU and call
the backend directly, forever, without the UI. Serving the app from a tier
that holds the key server-side is why `web/` exists. Smoke-test check 14
asserts the key is absent from `dist/` and `env-config.js`.

`API_KEY` must stay set in `backend/.env`: with it empty, `auth_check()` is a
no-op, and anything reaching 8010 directly on this shared host would have
unauthenticated access to the GPU.

---

## 10. Known quirks and gotchas

- **`AB_ENABLED` has an inverted default, in both halves.** Computed as
  `== "0"` in `backend/backend.py` *and* `frontend/src/config.ts`. So A/B mode
  is **ON unless the variable is set to something other than `"0"`**: unset =
  on, `AB_ENABLED=0` = on, `AB_ENABLED=1` = **off**. Almost certainly a bug,
  but it is how the app has been running, so the port preserved it exactly.
  Fixing it requires changing **both files together** or the two halves will
  disagree about what mode they are in.

- **`HF_ENDPOINT` vs `HF_HUB_ENDPOINT` — a genuine trap.**
  `huggingface_hub` reads `HF_ENDPOINT` as the *Hub* base URL. This app has
  used that same name since before RAG existed to mean the *LLM inference
  endpoint*. With RAG off the collision is invisible. Turn RAG on while
  `HF_ENDPOINT` points at local vLLM, and sentence-transformers tries to
  download the encoder **from the vLLM server**, 404ing on
  `{HF_ENDPOINT}/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json`.
  `_init_rag()` shadows `HF_ENDPOINT` with `HF_HUB_ENDPOINT` for the duration
  of the encoder load and restores it after. **Do not "simplify" that block.**

- **Pinecone client shape.** Pinned to `10.0.0`. v3+ returns a `QueryResponse`
  object, not a dict — `results.get("matches", [])` silently yields *nothing*
  on a current client. `_extract_matches()` accepts both shapes deliberately.

- **`RAG_ENABLED=1` without `requirements-rag.txt` installed** raises
  `ImportError` from `_init_rag()` on the first request. That is deliberate,
  not a silent fallback to no sources. `torch`, `sentence-transformers`,
  `numpy` and `pinecone` are imported *inside* `_init_rag()`, so the default
  slim install is ~20 packages and no CUDA wheels.

- **Node was unpacked into `/mnt/lagarza3/node`** from the official tarball —
  there is no system Node on this host and no root to add one. `web/server.js`
  is Node stdlib only: no dependencies, nothing to `npm install`, nothing to
  keep patched.

- **`web/server.js` reads `dist/` from disk per request**, so a frontend
  rebuild takes effect without restarting anything.

- **Superseded Cloudflare Containers path.** `frontend/Dockerfile`,
  `worker/index.ts` and `wrangler.toml` are still in the tree but are replaced
  by `web/`. They need a Workers Paid plan and carry the API-key-in-page-source
  problem. Left in place rather than deleted.

- **Hardcoded HF tokens in research code.** Several files under
  `llm-healthcare/` (`train/dpo_LLM.py`, `source_code/main.py`,
  `source_code/llm/*.py`, `source_code/Basic-RAG-healthcare-example.py`)
  contain a hardcoded Hugging Face token. These files are **untracked**, so
  the token was never pushed to the public GitHub repo — but it should be
  revoked and moved to an env var.

---

## 11. The A/B feature (a likely demo talking point)

`/generate_ab` sends the same conversation twice with two different system
prompts and returns both answers for the user to vote on:

- **Style A:** "Be concise and direct. Keep the response concise and do not
  exceed the allowed response length of 400 characters."
- **Style B:** "Be warm and empathic. Keep the response concise and do not
  exceed the allowed response length of 400 characters."

Votes go to `/ab_vote`, free-text feedback to `/comment`. This closes the loop
back to the training method: the app collects exactly the
concise-vs-empathic preference judgements that DPO consumes as
`{prompt, chosen, rejected}` triples. Both answers get their own Sources
block. A/B costs two full generations per turn (~4.9 s measured).

---

## 12. Likely demo questions, with honest answers

**Are the answers grounded in the cited sources?**
No. Citations are retrieved using the generated answer *after* generation and
appended. The model never sees them. It is post-hoc attribution.

**Why no sources on that answer?**
`RAG_MIN_SCORE=0.65` filtered everything out — the corpus does not cover that
question. Correct behaviour, not a bug.

**Is this safe for clinical use?**
No. It is a research prototype for caregiver support and information. There is
no clinical validation, no regulatory clearance, and no human-in-the-loop
review. The DPO training demonstrably improved tone and reading level; it was
not shown to improve clinical correctness.

**What did fine-tuning actually buy?**
Consistent gains on every metric and every base model, largest on empathy,
GEval and semantic similarity — plus a lower reading grade. The preference
pairs were easy to separate, so the honest claim is style, warmth and
accessibility rather than factual accuracy.

**Why Llama over Mistral/DeepSeek/Qwen?**
Best across the board after DPO. Mistral is a credible second and actually
edges Llama on empathy and NLI; DeepSeek starts far behind and gains the most
from DPO but does not catch up.

**Is patient data involved?**
No. The corpus is public material — alz.org, nia.nih.gov, YouTube caregiver
transcripts — plus public QA sets (MedQuAD, MedMCQA). No PHI.

**What would you do next?**
Turn post-hoc attribution into real context injection; get reading level from
~12 down toward grade 8; re-run the two suspect evaluation cells; fix the
inverted `AB_ENABLED` default; and use the collected A/B votes as a second
round of real human preference data instead of synthetic pairs.

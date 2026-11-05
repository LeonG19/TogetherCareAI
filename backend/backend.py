import os
import re
import json
import time
from threading import Lock
from typing import Any, Dict, List, Tuple, Optional

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import uuid

# Load backend/.env sitting next to this file. override=False so a real
# environment variable always wins over the file.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

# ---------------------- CONFIG ----------------------
HF_TOKEN = os.environ.get("HF_TOKEN")

API_KEY = os.environ.get("API_KEY")  # optional

LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_ENABLED = os.environ.get("LOG_ENABLED", "1") == "1"
_log_lock = Lock()

# NOTE: inverted default, preserved deliberately - A/B mode is ON unless this
# is set to something other than "0". frontend/src/config.ts has the same
# inversion. See README ("Known quirk: AB_ENABLED").
AB_ENABLED = os.environ.get("AB_ENABLED", "0") == "0"
AB_EVERY_N = int(os.environ.get("AB_EVERY_N", "5"))

COMMENTS_DIR = os.environ.get("COMMENTS_DIR", "comments")
COMMENTS_ENABLED = os.environ.get("COMMENTS_ENABLED", "1") == "1"

# Retrieval is opt-in. When it is off, torch / sentence-transformers / numpy /
# pinecone are never imported -- see _init_rag().
RAG_ENABLED = os.environ.get("RAG_ENABLED", "0") == "1"

# Any OpenAI-compatible /v1/chat/completions server: the hosted HF Inference
# Endpoint, or a local vLLM instance.
HF_ENDPOINT = os.environ.get(
    "HF_ENDPOINT",
    "https://x2cc012krs19t4x5.eu-west-1.aws.endpoints.huggingface.cloud",
).rstrip("/")
# Single source of truth for the served model name. This used to be hardcoded
# separately in _call_hf_endpoint() and in the /generate_stream route.
HF_MODEL = os.environ.get("HF_MODEL", "ERivas8/empathy-llama3-8b")
HF_TIMEOUT = float(os.environ.get("HF_TIMEOUT", "300"))

# Retry budget. An HF Inference Endpoint answers 503 while it wakes; a local
# vLLM server refuses the connection outright until it has finished loading
# weights. Both are survivable and both are retried. Defaults to ~2 minutes.
HF_MAX_RETRIES = int(os.environ.get("HF_MAX_RETRIES", "12"))
HF_RETRY_WAIT = float(os.environ.get("HF_RETRY_WAIT", "10"))

# Which retrieval store to use when RAG_ENABLED=1.
#   "faiss"    - local index directory, no API key, no network (default)
#   "pinecone" - hosted, needs PINECONE_API_KEY
RAG_BACKEND = os.environ.get("RAG_BACKEND", "faiss").strip().lower()

# A directory holding index.faiss + meta.jsonl + embed_model.txt, as produced
# by the llm-healthcare indexing scripts.
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "")

# Where huggingface_hub should fetch the sentence-transformers encoder from.
# Separate from HF_ENDPOINT above, which is this app's LLM inference endpoint
# and collides with the variable huggingface_hub reads -- see _init_rag().
# Set to "" to force offline / cache-only.
HF_HUB_ENDPOINT = os.environ.get("HF_HUB_ENDPOINT", "https://huggingface.co")

# Cosine-similarity floor for a chunk to be worth citing. The index is
# inner-product over L2-normalised vectors, so scores are cosine in [-1, 1].
RAG_MIN_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.7"))

# For RAG_BACKEND=pinecone. The key is never echoed anywhere -- the status
# route reports pinecone_api_key_set: true|false only, the way it already does
# for HF_TOKEN.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "empathy-rag")
# Empty string means the index's default namespace, which is where the
# empathy-rag vectors actually live (describe_index_stats reports the single
# namespace "__default__"). Set only if the index is later partitioned.
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# all-MiniLM-L6-v2 emits 384-dim vectors. Checked against the index's declared
# dimension at startup for the same reason FAISS checks embed_model.txt: a
# store built with a different encoder returns plausible nonsense, not an error.
EMBED_DIM = 384

# Maximum number of conversation turns to keep in context (1 turn = 1 user + 1 assistant message)
MAX_HISTORY_TURNS = 5

# Comma-separated list of allowed frontend origins for CORS, e.g.
#   ALLOWED_ORIGINS="https://togethercareai.example-worker.workers.dev,https://app.togethercare.ai"
# Defaults to "*" for initial testing; lock this down once the frontend's real
# Cloudflare Containers / custom domain URL is known.
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]


# ---------------------- APP ----------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------- RETRIEVAL STACK (LAZY, OPTIONAL) ----------------------
_embedder = None
_pinecone_index = None
# Vector count read once at startup from describe_index_stats(), so the status
# route can report it next to faiss_vectors without a Pinecone call per probe.
_pinecone_vectors: Optional[int] = None
_faiss_index = None
_faiss_meta: List[Dict[str, Any]] = []
_rag_ready = False


def _init_rag() -> None:
    """Import and construct the retrieval stack.

    Only ever reached when RAG_ENABLED is true. torch, sentence-transformers,
    numpy and pinecone are imported here rather than at module scope so that
    LLM-only deployments neither need a Pinecone key nor drag in torch.
    """
    global _embedder, _pinecone_index, _pinecone_vectors
    global _faiss_index, _faiss_meta, _rag_ready
    if _rag_ready:
        return

    # Config-only checks first. Loading the encoder costs ~6s and ~1 GB, and
    # there is no reason to pay that before finding out the store it is meant
    # to query was never configured.
    if RAG_BACKEND == "pinecone" and not PINECONE_API_KEY:
        raise RuntimeError("RAG_BACKEND=pinecone but PINECONE_API_KEY is not set")
    if RAG_BACKEND == "faiss" and not FAISS_INDEX_DIR:
        raise RuntimeError("RAG_BACKEND=faiss but FAISS_INDEX_DIR is not set")
    if RAG_BACKEND not in ("faiss", "pinecone"):
        raise RuntimeError(
            f"unknown RAG_BACKEND '{RAG_BACKEND}' (expected 'faiss' or 'pinecone')"
        )

    # ---- HF_ENDPOINT name collision, read this before "simplifying" it ----
    # huggingface_hub treats the environment variable HF_ENDPOINT as the Hub
    # base URL. This app has used the same name since before RAG existed, to
    # mean "the LLM inference endpoint" -- which right now is a local vLLM
    # server. Left alone, sentence-transformers tries to download the encoder
    # from the vLLM server and dies with a 404 on
    #   {HF_ENDPOINT}/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
    # The collision is invisible with RAG off because huggingface_hub is never
    # imported. Shadow the variable for the duration of the load, then put it
    # back so _hf_post() keeps talking to the right place.
    _saved_hf_endpoint = os.environ.pop("HF_ENDPOINT", None)
    if HF_HUB_ENDPOINT:
        os.environ["HF_ENDPOINT"] = HF_HUB_ENDPOINT
    try:
        from sentence_transformers import SentenceTransformer

        # device="cpu" is deliberate and load-bearing on this host: the GPU is
        # shared and vLLM has already claimed nearly all of the free VRAM. A
        # 22M-parameter MiniLM encoder is fast enough on CPU.
        print(f"[backend] RAG_ENABLED=1 backend={RAG_BACKEND} - loading embedder...")
        _embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    finally:
        os.environ.pop("HF_ENDPOINT", None)
        if _saved_hf_endpoint is not None:
            os.environ["HF_ENDPOINT"] = _saved_hf_endpoint

    if RAG_BACKEND == "faiss":
        import faiss

        d = Path(FAISS_INDEX_DIR)
        idx_path, meta_path = d / "index.faiss", d / "meta.jsonl"
        for f in (idx_path, meta_path):
            if not f.exists():
                raise RuntimeError(f"FAISS_INDEX_DIR is missing {f.name}: {f}")

        # The index was built for a specific encoder. Using a different one
        # silently returns garbage neighbours rather than failing, so check.
        stamp = d / "embed_model.txt"
        if stamp.exists():
            want = stamp.read_text().strip()
            if want and want != EMBED_MODEL_NAME:
                raise RuntimeError(
                    f"index was built with '{want}' but EMBED_MODEL_NAME is "
                    f"'{EMBED_MODEL_NAME}'; embeddings would not be comparable"
                )

        print(f"[backend] loading FAISS index {idx_path}...")
        _faiss_index = faiss.read_index(str(idx_path))

        print(f"[backend] loading metadata {meta_path}...")
        _faiss_meta = []
        with open(meta_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    _faiss_meta.append(json.loads(line))

        if _faiss_index.ntotal != len(_faiss_meta):
            raise RuntimeError(
                f"index/metadata mismatch: {_faiss_index.ntotal} vectors vs "
                f"{len(_faiss_meta)} metadata rows -- they must line up, since "
                f"FAISS returns positions, not ids"
            )

        _rag_ready = True
        print(
            f"[backend] RAG ready. faiss vectors={_faiss_index.ntotal} "
            f"dim={_faiss_index.d} metric="
            f"{'IP' if _faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT else 'L2'}"
        )

    elif RAG_BACKEND == "pinecone":
        from pinecone import Pinecone

        # Both calls are inside the try: the v10 client resolves the index
        # host inside pc.Index(), so a wrong PINECONE_INDEX_NAME 404s and a bad
        # key 401s at that line rather than at the stats call.
        #
        # describe_index_stats() then does two more jobs: it confirms the index
        # is actually readable and it yields the dimension. Doing this at
        # startup matters because retrieve_top_k() is wrapped in try/except at
        # every call site -- a store that only fails on first use would have
        # its error printed to the log and swallowed, leaving a chatbot that
        # answers normally and silently never cites anything.
        try:
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index = pc.Index(PINECONE_INDEX_NAME)
            stats = index.describe_index_stats()
        except Exception as e:
            raise RuntimeError(
                f"could not reach Pinecone index '{PINECONE_INDEX_NAME}' "
                f"({e.__class__.__name__}: {e}); check PINECONE_API_KEY and "
                f"PINECONE_INDEX_NAME"
            ) from e

        dim = getattr(stats, "dimension", None)
        if dim is None and isinstance(stats, dict):
            dim = stats.get("dimension")
        if dim is not None and int(dim) != EMBED_DIM:
            # The FAISS branch catches this via embed_model.txt. Pinecone
            # carries no encoder stamp, so dimension is the only automatic
            # check available -- it catches the whole-wrong-encoder case,
            # though not a different 384-dim model.
            raise RuntimeError(
                f"pinecone index '{PINECONE_INDEX_NAME}' has dimension {dim} "
                f"but EMBED_MODEL_NAME '{EMBED_MODEL_NAME}' produces "
                f"{EMBED_DIM}-dim vectors; embeddings would not be comparable"
            )

        _pinecone_index = index
        _rag_ready = True

        total = getattr(stats, "total_vector_count", None)
        if total is None and isinstance(stats, dict):
            total = stats.get("total_vector_count")
        _pinecone_vectors = int(total) if total is not None else None
        print(
            f"[backend] RAG ready. pinecone_index={PINECONE_INDEX_NAME} "
            f"vectors={total} dim={dim} "
            f"namespace={PINECONE_NAMESPACE or '<default>'}"
        )

    else:
        raise RuntimeError(
            f"unknown RAG_BACKEND '{RAG_BACKEND}' (expected 'faiss' or 'pinecone')"
        )


if RAG_ENABLED:
    _init_rag()
    _mode_note = (
        f"RAG mode (backend={RAG_BACKEND}, "
        + (f"index_dir={FAISS_INDEX_DIR}" if RAG_BACKEND == "faiss"
           else f"pinecone_index={PINECONE_INDEX_NAME} "
                f"namespace={PINECONE_NAMESPACE or '<default>'}")
        + ")"
    )
else:
    _mode_note = "LLM-only mode (RAG_ENABLED=0; torch/pinecone not imported)"

print(f"[backend] ready. {_mode_note}, hf_endpoint={HF_ENDPOINT}, hf_model={HF_MODEL}")


# ---------------------- PROMPTS ----------------------
# Every system prompt sent to the model is built by _with_base_prompt(), so
# the A/B styles and the single-style routes all inherit the same audience and
# glossary blocks. Edit the blocks here, not at the call sites.
SHORT_INSTRUCTION = "Keep the response very short: at most 3 sentences, with no lists or headings."

AUDIENCE_BLOCK = (
    "AUDIENCE: Assume the person writing to you is a family caregiver, a professional caregiver, "
    "or a health professional supporting someone with Alzheimer's disease or a related dementia. "
    "Do not assume the user is the patient unless they say so. Refer to the person being cared for "
    "as 'your loved one' or 'the person you care for' unless the user uses another term. "
    "Keep a person-centered approach."
)

GLOSSARY_BLOCK = """TERMINOLOGY: Interpret these terms in a dementia caregiving context.
- ADRD: Alzheimer's disease and related dementias. This is NOT ADHD.
- Dementia: an umbrella term for a decline in memory and thinking that affects daily life. Alzheimer's disease is the most common cause.
- MCI: mild cognitive impairment.
- LBD: Lewy body dementia. FTD: frontotemporal dementia.
- BPSD: behavioral and psychological symptoms of dementia, such as agitation, wandering, and aggression.
- ADLs: activities of daily living such as bathing, dressing, eating, and transferring. IADLs: instrumental activities such as managing money, medications, and transportation.
- PCP: primary care provider. PT: physical therapist. OT: occupational therapist.
- The 4 Ms: the Age-Friendly Health Systems framework of What Matters, Medication, Mentation, and Mobility.
If the user uses an acronym you do not recognize, do not silently map it to a different medical condition. Say how you are interpreting it, or ask what they mean."""


DETAILED_INSTRUCTION = (
    "Give a thorough, well-organized response with practical steps the caregiver can take. "
    "Use short paragraphs or bullet points and keep it under about 250 words."
)

# Response length chosen by the frontend toggle. Anything unrecognised falls
# back to "short".
#
# The served model is a fine-tune that writes long answers whatever the prompt
# says (tested: system instructions, worked examples and a rewrite pass all
# left "short" answers at 250-400 words). What does work is the combination
# below: the instruction, a reminder appended to the model's copy of the user
# message, a small max_tokens budget, and cutting any unfinished final sentence
# (_trim_to_sentence), which gives roughly 80-125 words of complete sentences.
MODE_INSTRUCTIONS = {"short": SHORT_INSTRUCTION, "detailed": DETAILED_INSTRUCTION}
MODE_MAX_TOKENS = {"short": 150, "detailed": 768}
SHORT_REMINDER = "\n\n(Answer in at most 3 sentences, with no lists or headings.)"
DEFAULT_MODE = "short"


def _with_base_prompt(instruction: str) -> str:
    return "\n\n".join([instruction, AUDIENCE_BLOCK, GLOSSARY_BLOCK])


# Full system prompt per mode (both inherit the audience and glossary blocks).
SYSTEM_PROMPTS = {m: _with_base_prompt(i) for m, i in MODE_INSTRUCTIONS.items()}


def _mode_of(req: "GenerateRequest") -> str:
    return req.mode if req.mode in MODE_INSTRUCTIONS else DEFAULT_MODE


def _max_tokens(req: "GenerateRequest") -> int:
    """Token budget is owned by the mode; the request's max_new_tokens is ignored."""
    return MODE_MAX_TOKENS[_mode_of(req)]


def _with_length_reminder(messages: List[Dict[str, str]], mode: str) -> List[Dict[str, str]]:
    """Copy of messages with the short-mode reminder on the final user turn.
    Only the model sees it: the displayed message, history and logs don't."""
    if mode != "short" or not messages or messages[-1].get("role") != "user":
        return messages
    return messages[:-1] + [{**messages[-1], "content": messages[-1]["content"] + SHORT_REMINDER}]


_SENTENCE_END = re.compile(r'[.!?]["\')\]]?(?=\s|$)')


def _trim_to_sentence(text: str) -> str:
    """Drop an unfinished final sentence (used when the model hit max_tokens).
    Text with no sentence end at all is returned unchanged."""
    ends = list(_SENTENCE_END.finditer(text))
    return text[: ends[-1].end()].rstrip() if ends else text


def _retrieve_for_turn(message: str, answer: str, k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Retrieve sources for one turn using two queries and keep each document's
    best score: the question + answer, and the question alone. Neither wins
    everywhere (the question alone is better for definitions, the answer adds
    context for vague questions), and short answers carry little topical text,
    so a single query often falls just under RAG_MIN_SCORE. The question goes
    first in the combined text because the embedder truncates at 256 tokens and
    replies open with sympathy."""
    best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
    for query in (f"{message}\n{answer}", message):
        for score, row in retrieve_top_k(query, k=k):
            key = _dedup_key(extract_url(row)) or str(id(row))
            if key not in best or score > best[key][0]:
                best[key] = (score, row)
    return sorted(best.values(), key=lambda t: t[0], reverse=True)


# Pure greetings, thanks and "I'm fine" replies get a fixed answer instead of a
# model call. The fine-tune treats any message as a distressed caregiver and
# answers "hello there" with a long monologue (which then also pulls in sources
# that have nothing to do with the question). No model, no retrieval.
_GREETING_RE = re.compile(
    r"(hi|hello|hey|hiya|howdy|good (morning|afternoon|evening))( there| everyone)?"
    r"|how are you( today| doing)?"
    r"|i( am|'m|m) (good|fine|ok|okay|well|great|doing (good|fine|well|okay))( thanks| thank you)?"
)
_THANKS_RE = re.compile(r"(thanks|thank you|thx|ty)( so much| a lot| very much)?( again)?|bye|goodbye")
GREETING_REPLY = (
    "Hello! I'm TogetherCare AI. I can answer questions about caring for someone with "
    "Alzheimer's disease or a related dementia. What would you like to know?"
)
THANKS_REPLY = "You're welcome! Let me know if there's anything else I can help with."


def _canned_reply(message: str) -> Optional[str]:
    """A fixed reply if the whole message is a greeting / thanks, else None."""
    text = re.sub(r"[^a-z' ]+", " ", message.lower().replace("\u2019", "'")).strip()
    text = re.sub(r"\s+", " ", text)
    if _THANKS_RE.fullmatch(text):
        return THANKS_REPLY
    return GREETING_REPLY if _GREETING_RE.fullmatch(text) else None


# Retrieval-only acronym expansion (see expand_acronyms). Ambiguous short
# terms such as "AD" are deliberately left out.
ACRONYM_EXPANSIONS = {
    "ADRD": "Alzheimer's disease and related dementias",
    "MCI": "mild cognitive impairment",
    "LBD": "Lewy body dementia",
    "FTD": "frontotemporal dementia",
    "BPSD": "behavioral and psychological symptoms of dementia",
    "ADLs": "activities of daily living",
    "IADLs": "instrumental activities of daily living",
}
# Whole word, case-insensitive; the lookahead skips terms that are already
# followed by a parenthetical so an expanded query is never expanded twice.
_ACRONYM_RE = re.compile(
    r"\b(" + "|".join(map(re.escape, ACRONYM_EXPANSIONS)) + r")\b(?!\s*\()", re.IGNORECASE
)
_ACRONYM_LOOKUP = {k.lower(): v for k, v in ACRONYM_EXPANSIONS.items()}


def expand_acronyms(text: str) -> str:
    """'What is ADRD?' -> "What is ADRD (Alzheimer's disease and related dementias)?".
    For the retrieval query only: never apply this to text shown to the user
    or sent to the LLM."""
    return _ACRONYM_RE.sub(lambda m: f"{m.group(1)} ({_ACRONYM_LOOKUP[m.group(1).lower()]})", text)


# ---------------------- HELPERS ----------------------
def extract_url(meta_row: dict) -> str:
    m = meta_row.get("meta", {})
    if isinstance(m, str):
        return m
    for key in ["url", "source", "link", "href", "document_url", "source_url"]:
        if isinstance(m, dict) and m.get(key):
            return m[key]
    for key in ["url", "source", "link"]:
        if meta_row.get(key):
            return meta_row[key]
    return ""


def _dedup_key(url: str) -> str:
    """Collapse the trivial URL variants that would otherwise be cited twice.

    The Pinecone corpus stores the same nia.nih.gov page under both http:// and
    https://, so an exact-string `seen` set treats them as two sources and a
    3-slot Sources block spends two slots on one article. Normalise scheme,
    a leading "www.", the host's case and a trailing slash for comparison
    only -- the URL actually shown to the user is still the one from the
    metadata, untouched. Different paths stay different.
    """
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()

    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/")
    # Scheme is deliberately dropped, not normalised: http and https of the
    # same page are the duplicate this exists to catch.
    tail = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{tail}" if host else url.strip().lower()


def embed_text(text: str):
    import numpy as np  # RAG-only dependency; see _init_rag()

    _init_rag()
    vec = _embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


def _extract_matches(results: Any) -> List[Dict[str, Any]]:
    """Normalise a Pinecone query response into a list of plain dicts.

    Pinecone changed the response type in v3: pre-v3 clients hand back a dict
    that supports .get("matches") and whose entries are dicts, current clients
    hand back a QueryResponse object whose matches are attribute-accessed and
    whose entries are ScoredVector objects. results.get("matches", []) silently
    yields nothing on a current client, so accept both shapes.
    """
    matches = None
    if isinstance(results, dict):
        matches = results.get("matches")
    if matches is None:
        matches = getattr(results, "matches", None)
    if matches is None and hasattr(results, "__getitem__"):
        try:
            matches = results["matches"]
        except (KeyError, TypeError):
            matches = None

    out: List[Dict[str, Any]] = []
    for m in matches or []:
        if isinstance(m, dict):
            out.append(m)
        elif hasattr(m, "to_dict"):
            out.append(m.to_dict())
        else:
            out.append(
                {
                    "score": getattr(m, "score", None),
                    "metadata": getattr(m, "metadata", None),
                }
            )
    return out


def _retrieve_faiss(query_text: str, k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Search the local FAISS index.

    The index is inner-product over L2-normalised vectors, and embed_text()
    normalises too, so the returned scores are cosine similarities directly
    comparable to RAG_MIN_SCORE.
    """
    q = embed_text(query_text)  # (1, dim), already normalised
    scores, positions = _faiss_index.search(q, k)

    out: List[Tuple[float, Dict[str, Any]]] = []
    for score, pos in zip(scores[0], positions[0]):
        # FAISS pads with -1 when it finds fewer than k neighbours.
        if pos < 0 or pos >= len(_faiss_meta):
            continue
        rec = _faiss_meta[pos]
        meta = rec.get("meta") or {}
        # Flatten title/url/source to the top level so extract_url(),
        # _title_from_row() and format_sources() see the same shape they get
        # from a Pinecone metadata dict, while keeping the nested "meta" key
        # that extract_url() checks first.
        row = {
            **(meta if isinstance(meta, dict) else {}),
            "id": rec.get("id"),
            "path": rec.get("path"),
            "text": rec.get("text"),
            "meta": meta,
        }
        out.append((float(score), row))
    return out


def _retrieve_pinecone(query_text: str, k: int) -> List[Tuple[float, Dict[str, Any]]]:
    """Search the hosted Pinecone index.

    The empathy-rag index is metric="cosine" and embed_text() L2-normalises,
    so the scores come back on the same [-1, 1] cosine scale the FAISS branch
    produces and are directly comparable to RAG_MIN_SCORE. Metadata rows carry
    title/url/text/source_id, which is the flat shape extract_url(),
    _title_from_row() and format_sources() already expect.
    """
    q = embed_text(query_text)[0].tolist()

    kwargs: Dict[str, Any] = {"vector": q, "top_k": k, "include_metadata": True}
    # Only send the argument when one is configured: passing namespace="" is
    # the documented way to address the default namespace, but leaving it out
    # entirely is what older clients expect, and the vectors live in the
    # default namespace either way.
    if PINECONE_NAMESPACE:
        kwargs["namespace"] = PINECONE_NAMESPACE

    results = _pinecone_index.query(**kwargs)

    out: List[Tuple[float, Dict[str, Any]]] = []
    for match in _extract_matches(results):
        score = match.get("score")
        # A match with no score at all is a malformed response, not a good
        # hit. It used to default to 0.7, which sails past the default
        # RAG_MIN_SCORE and gets cited; 0.0 makes it fall below any sane
        # threshold instead, so a broken response cites nothing rather than
        # citing something unranked.
        score = float(score) if score is not None else 0.0
        meta = match.get("metadata") or {}
        out.append((score, meta))
    return out


def retrieve_top_k(query_text: str, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    if not RAG_ENABLED:
        return []

    _init_rag()
    query_text = expand_acronyms(query_text)

    if RAG_BACKEND == "faiss":
        return _retrieve_faiss(query_text, k)
    return _retrieve_pinecone(query_text, k)


def format_sources(
    results: List[Tuple[float, Dict[str, Any]]],
    max_sources: int = 3,
    min_score: Optional[float] = None,
) -> str:
    """Render a Sources block. min_score defaults to RAG_MIN_SCORE so the
    env var actually takes effect -- the four call sites do not pass it."""
    if min_score is None:
        min_score = RAG_MIN_SCORE
    seen = set()
    lines = []
    added = 0
    for score, row in results:
        if score < min_score:
            continue
        url = extract_url(row)
        key = _dedup_key(url)
        if not url or key in seen:
            continue
        seen.add(key)
        title = row.get("title") or row.get("id") or "source"
        lines.append(f"- {title}: {url}")
        added += 1
        if added >= max_sources:
            break
    return "\n\n**Sources:**\n" + "\n".join(lines) if lines else ""


def auth_check(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _safe_name(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = "".join(c for c in s if c.isalnum() or c in ("_", "-"))
    return s[:64]


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_session_id(s: str) -> str:
    s = (s or "").strip()
    s = "".join(c for c in s if c.isalnum() or c in ("_", "-"))
    return s[:128]


def _title_from_row(row: Dict[str, Any]) -> str:
    """Best available human label for a retrieved chunk.

    "title" comes first to match format_sources(), which the user actually
    sees. Pinecone metadata carries title/url/text/source_id and no id or
    path, so without the title fallback every structured log row from the
    Pinecone backend was labelled "source".
    """
    return row.get("title") or row.get("id") or row.get("path") or "source"


def _structured_retrieval(results: List[Tuple[float, Dict[str, Any]]], max_sources: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for score, row in results:
        url = extract_url(row)
        key = _dedup_key(url)
        if not url or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "score": float(score),
                "title": _title_from_row(row),
                "url": url,
            }
        )
        if len(out) >= max_sources:
            break
    return out


def _truncate_history(history: List[Dict[str, str]], max_turns: int = MAX_HISTORY_TURNS) -> List[Dict[str, str]]:
    """Keep only the most recent max_turns exchanges to stay within the context limit.
    Each turn = 1 user message + 1 assistant message = 2 entries in history.
    """
    if len(history) > max_turns * 2:
        return history[-(max_turns * 2):]
    return history


def log_interaction_jsonl(
    *,
    session_id: str,
    endpoint: str,
    mode: str,
    req: "GenerateRequest",
    answer_text: str,
    retrieval_struct: Optional[List[Dict[str, Any]]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    if not LOG_ENABLED:
        return

    sid = _safe_session_id(session_id)
    if not sid:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")
        return

    _ensure_dir(LOG_DIR)
    log_path = os.path.join(LOG_DIR, f"{sid}.jsonl")

    record = {
        "session_id": sid,
        "interaction_id": uuid.uuid4().hex,
        "timestamp_utc": _utc_iso(),
        "endpoint": endpoint,
        "mode": mode,
        "request": {
            "message": req.message,
            "mode": _mode_of(req),
            "gen_params": {
                "max_new_tokens": int(req.max_new_tokens),
                "temperature": float(req.temperature),
                "top_p": float(req.top_p),
                "top_k_sources": int(req.top_k_sources),
            },
        },
        "response": {
            "answer_text": answer_text,
        },
        "retrieval": {
            "top_k": int(req.top_k_sources),
            "results": retrieval_struct or [],
        },
        "error": error,
    }

    line = json.dumps(record, ensure_ascii=False)
    with _log_lock:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _hf_post(payload: Dict[str, Any], stream: bool = False) -> requests.Response:
    """POST to the OpenAI-compatible /v1/chat/completions route, with retries.

    Two distinct transient failures are retried:
      * HTTP 503 - an HF Inference Endpoint that is scaled to zero and waking.
      * ConnectionError - a local vLLM server that has not finished loading
        weights yet, so nothing is listening on the port and the connection is
        refused outright rather than answered with a status code.

    Anything else (401, 404 for a wrong model name, 422) raises immediately;
    those are configuration errors and retrying only hides them.
    """
    url = HF_ENDPOINT + "/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }

    last_reason = "no attempt made"
    for attempt in range(1, HF_MAX_RETRIES + 1):
        try:
            response = requests.post(
                url, headers=headers, json=payload, stream=stream, timeout=HF_TIMEOUT
            )
        except requests.exceptions.ConnectionError as e:
            last_reason = f"connection refused ({e.__class__.__name__})"
        else:
            if response.status_code != 503:
                response.raise_for_status()
                return response
            last_reason = "HTTP 503 (endpoint waking)"
            response.close()

        if attempt < HF_MAX_RETRIES:
            # flush=True: stdout is block-buffered when uvicorn's output is
            # redirected to a file, so without it these lines only appear once
            # the buffer fills -- i.e. long after the retries they describe,
            # which defeats the point of logging them.
            print(
                f"[backend][hf] attempt {attempt}/{HF_MAX_RETRIES} to {url} failed: "
                f"{last_reason}; retrying in {HF_RETRY_WAIT:g}s",
                flush=True,
            )
            time.sleep(HF_RETRY_WAIT)

    raise RuntimeError(
        f"{HF_ENDPOINT} did not become available after {HF_MAX_RETRIES} attempts "
        f"over ~{HF_MAX_RETRIES * HF_RETRY_WAIT:g}s (last failure: {last_reason})"
    )


def _call_hf_endpoint(messages: List[Dict[str, str]], max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Call the LLM endpoint via OpenAI-compatible API and return the generated text."""
    response = _hf_post(
        {
            "model": HF_MODEL,
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
    )
    choice = response.json()["choices"][0]
    text = choice["message"]["content"]
    return _trim_to_sentence(text) if choice.get("finish_reason") == "length" else text


def log_comment_jsonl(*, session_id: str, interaction_id: str, stars: int, comment: str) -> None:
    if not COMMENTS_ENABLED:
        return

    sid = _safe_session_id(session_id)
    if not sid:
        raise HTTPException(status_code=400, detail="Missing sesh_id header")

    if not interaction_id:
        raise HTTPException(status_code=400, detail="Missing interaction_id")

    if not (1 <= int(stars) <= 5):
        raise HTTPException(status_code=400, detail="stars must be between 1 and 5")

    _ensure_dir(COMMENTS_DIR)
    path = os.path.join(COMMENTS_DIR, f"{sid}.jsonl")

    record = {
        "session_id": sid,
        "interaction_id": interaction_id,
        "timestamp_utc": _utc_iso(),
        "stars": int(stars),
        "comment": (comment or "").strip(),
    }

    with _log_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------- API SCHEMAS ----------------------
class GenerateRequest(BaseModel):
    message: str
    history: List[Dict[str, str]] = []
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k_sources: int = 3
    mode: str = DEFAULT_MODE  # "short" | "detailed"


class GenerateResponse(BaseModel):
    answer: str


# ---------------------- OTHER CLASSES ----------------------
class ABVoteRequest(BaseModel):
    interaction_id: str
    choice: str  # "A" or "B"


class CommentRequest(BaseModel):
    interaction_id: str
    stars: int  # 1..5
    comment: Optional[str] = ""


# ---------------------- ROUTES ----------------------
def _endpoint_error_detail(e: Exception) -> str:
    """Turn an upstream failure into a message that names the endpoint and the
    model, without ever leaking the token."""
    if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
        body = (e.response.text or "")[:300]
        return (
            f"{HF_ENDPOINT} returned HTTP {e.response.status_code} for model "
            f"'{HF_MODEL}': {body}"
        )
    return f"{HF_ENDPOINT} (model '{HF_MODEL}') unreachable: {e}"


# Named /togethercareAI rather than /health at Dr. Pal's request. The two
# aliases below keep the old paths answering so that anything already pointed
# at them -- the Cloudflare tunnel, an uptime monitor, a bookmarked curl --
# does not start 404ing on deploy. They are hidden from the OpenAPI schema so
# the docs show one name.
@app.get("/togethercareAI")
def togethercare_ai():
    """Config echo: what this backend is wired to. Never returns a secret."""
    is_pinecone = RAG_ENABLED and RAG_BACKEND == "pinecone"
    return {
        "ok": True,
        "service": "togethercareAI",
        "rag_enabled": RAG_ENABLED,
        "hf_endpoint": HF_ENDPOINT,
        "hf_model": HF_MODEL,
        "hf_token_set": bool(HF_TOKEN),
        "rag_backend": RAG_BACKEND if RAG_ENABLED else None,
        # Only report the store actually in use -- reporting a pinecone index
        # name while serving from a local FAISS directory sends anyone
        # debugging this to the wrong place.
        "faiss_index_dir": FAISS_INDEX_DIR if (RAG_ENABLED and RAG_BACKEND == "faiss") else None,
        "faiss_vectors": (_faiss_index.ntotal if _faiss_index is not None else None),
        "pinecone_index": PINECONE_INDEX_NAME if is_pinecone else None,
        "pinecone_namespace": (PINECONE_NAMESPACE or "__default__") if is_pinecone else None,
        # Same treatment as hf_token_set: presence, never the value.
        "pinecone_api_key_set": bool(PINECONE_API_KEY) if is_pinecone else None,
        "pinecone_vectors": _pinecone_vectors if is_pinecone else None,
    }


@app.get("/health", include_in_schema=False)
def health():
    """Deprecated alias for /togethercareAI. Kept so existing probes survive."""
    return togethercare_ai()


@app.get("/togethercareAI/llm")
def togethercare_ai_llm(x_api_key: Optional[str] = Header(default=None)):
    """One small real generation against the configured endpoint.

    Lets a sleeping/loading endpoint be told apart from a bad token or a wrong
    model name without opening the UI: a wrong name 404s here, a bad token
    401s, and a loading server retries and then reports as unreachable.

    Unlike /togethercareAI this one burns GPU time, so it is behind
    auth_check(): once the service is reachable from outside the host, an
    unauthenticated route that runs inference is an open compute endpoint even
    with a fixed prompt and a 16-token cap. /togethercareAI stays open so
    liveness can be checked without a key.
    """
    auth_check(x_api_key)
    try:
        text = _call_hf_endpoint(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            max_new_tokens=16,
            temperature=0.0,
            top_p=1.0,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=_endpoint_error_detail(e))

    return {
        "ok": True,
        "hf_endpoint": HF_ENDPOINT,
        "hf_model": HF_MODEL,
        "text": text,
    }


@app.get("/health/llm", include_in_schema=False)
def health_llm(x_api_key: Optional[str] = Header(default=None)):
    """Deprecated alias for /togethercareAI/llm."""
    return togethercare_ai_llm(x_api_key)


@app.post("/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    x_api_key: Optional[str] = Header(default=None),
    x_session_id: Optional[str] = Header(default=None, alias="sesh_id"),
):
    auth_check(x_api_key)

    canned = _canned_reply(req.message)
    if canned:
        try:
            log_interaction_jsonl(session_id=x_session_id or "", endpoint="/generate", mode="canned", req=req, answer_text=canned)
        except Exception as e:
            print(f"[backend][log] failed: {e}")
        return {"answer": canned}

    messages = _truncate_history(list(req.history) if req.history else [])
    messages.append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[_mode_of(req)]}] + _with_length_reminder(messages, _mode_of(req))

    try:
        answer = _call_hf_endpoint(messages, _max_tokens(req), req.temperature, req.top_p)
    except Exception as e:
        raise HTTPException(status_code=502, detail=_endpoint_error_detail(e))

    retrieval_struct = []
    sources = ""
    try:
        results = _retrieve_for_turn(req.message, answer, k=int(req.top_k_sources))
        sources = format_sources(results, max_sources=int(req.top_k_sources))
        retrieval_struct = _structured_retrieval(results, max_sources=int(req.top_k_sources))
        if sources:
            answer += sources
    except Exception as e:
        print(f"[backend][retrieval] failed: {e}")

    try:
        log_interaction_jsonl(
            session_id=x_session_id or "",
            endpoint="/generate",
            mode="hf_endpoint",
            req=req,
            answer_text=answer,
            retrieval_struct=retrieval_struct,
        )
    except Exception as e:
        print(f"[backend][log] failed: {e}")

    return {"answer": answer}


@app.post("/generate_ab")
def generate_ab(
    req: GenerateRequest,
    x_api_key: Optional[str] = Header(default=None),
    x_session_id: Optional[str] = Header(default=None, alias="sesh_id"),
):
    auth_check(x_api_key)

    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")

    base_messages = _truncate_history(list(req.history) if req.history else [])
    base_messages.append({"role": "user", "content": req.message})

    mode_instruction = MODE_INSTRUCTIONS[_mode_of(req)]
    style_a = f"Be concise and direct. {mode_instruction}"
    style_b = f"Be warm and empathic. {mode_instruction}"

    model_messages = _with_length_reminder(base_messages, _mode_of(req))
    messages_a = [{"role": "system", "content": _with_base_prompt(style_a)}] + model_messages
    messages_b = [{"role": "system", "content": _with_base_prompt(style_b)}] + model_messages

    answer_a = _call_hf_endpoint(messages_a, _max_tokens(req), req.temperature, req.top_p)
    answer_b = _call_hf_endpoint(messages_b, _max_tokens(req), req.temperature, req.top_p)

    sources_a = ""
    sources_b = ""
    retr_a = []
    retr_b = []

    try:
        res_a = _retrieve_for_turn(req.message, answer_a, k=int(req.top_k_sources))
        sources_a = format_sources(res_a, max_sources=int(req.top_k_sources))
        retr_a = _structured_retrieval(res_a, max_sources=int(req.top_k_sources))
        if sources_a:
            answer_a += sources_a
    except Exception as e:
        print(f"[backend][ab][retrieval A] failed: {e}")

    try:
        res_b = _retrieve_for_turn(req.message, answer_b, k=int(req.top_k_sources))
        sources_b = format_sources(res_b, max_sources=int(req.top_k_sources))
        retr_b = _structured_retrieval(res_b, max_sources=int(req.top_k_sources))
        if sources_b:
            answer_b += sources_b
    except Exception as e:
        print(f"[backend][ab][retrieval B] failed: {e}")

    interaction_id = uuid.uuid4().hex

    try:
        log_interaction_jsonl(
            session_id=x_session_id,
            endpoint="/generate_ab",
            mode="hf_endpoint",
            req=req,
            answer_text="",
            retrieval_struct=[],
            error=None,
        )
        ab_record = {
            "type": "ab_generation",
            "session_id": _safe_session_id(x_session_id),
            "interaction_id": interaction_id,
            "timestamp_utc": _utc_iso(),
            "mode": "hf_endpoint",
            "request": {
                "message": req.message,
                "gen_params": {
                    "max_new_tokens": int(req.max_new_tokens),
                    "temperature": float(req.temperature),
                    "top_p": float(req.top_p),
                    "top_k_sources": int(req.top_k_sources),
                },
            },
            "ab": {
                "style_a": style_a,
                "style_b": style_b,
                "answer_a": answer_a,
                "answer_b": answer_b,
                "retrieval_a": retr_a,
                "retrieval_b": retr_b,
                "chosen": None,
            },
        }
        _ensure_dir(LOG_DIR)
        log_path = os.path.join(LOG_DIR, f"{_safe_session_id(x_session_id)}.jsonl")
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ab_record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[backend][ab][log] failed: {e}")

    return {
        "interaction_id": interaction_id,
        "answer_a": answer_a,
        "answer_b": answer_b,
    }


@app.post("/ab_vote")
def ab_vote(
    vote: ABVoteRequest,
    x_api_key: Optional[str] = Header(default=None),
    x_session_id: Optional[str] = Header(default=None, alias="sesh_id"),
):
    auth_check(x_api_key)

    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing X-Session-Id header")

    choice = (vote.choice or "").strip().upper()
    if choice not in ("A", "B"):
        raise HTTPException(status_code=400, detail="choice must be 'A' or 'B'")

    try:
        record = {
            "type": "ab_vote",
            "session_id": _safe_session_id(x_session_id),
            "interaction_id": vote.interaction_id,
            "timestamp_utc": _utc_iso(),
            "choice": choice,
        }
        _ensure_dir(LOG_DIR)
        log_path = os.path.join(LOG_DIR, f"{_safe_session_id(x_session_id)}.jsonl")
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[backend][ab_vote][log] failed: {e}")

    return {"ok": True}


@app.post("/generate_stream")
def generate_stream(
    req: GenerateRequest,
    x_api_key: Optional[str] = Header(default=None),
    x_session_id: Optional[str] = Header(default=None, alias="sesh_id"),
):
    auth_check(x_api_key)

    canned = _canned_reply(req.message)
    if canned:
        def canned_stream():
            try:
                log_interaction_jsonl(session_id=x_session_id or "", endpoint="/generate_stream", mode="canned", req=req, answer_text=canned)
            except Exception as e:
                print(f"[backend][log] failed: {e}")
            yield f"data: {json.dumps({'token': canned})}\n\n"
            yield f"data: {json.dumps({'event': 'meta', 'interaction_id': uuid.uuid4().hex})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(canned_stream(), media_type="text/event-stream")

    messages = _truncate_history(list(req.history) if req.history else [])
    messages.append({"role": "user", "content": req.message})
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[_mode_of(req)]}] + _with_length_reminder(messages, _mode_of(req))

    def event_generator():
        interaction_id = uuid.uuid4().hex

        final_text = ""
        sources = ""
        retrieval_struct = []
        error_obj = None

        try:
            response = _hf_post(
                {
                    "model": HF_MODEL,
                    "messages": messages,
                    "max_tokens": _max_tokens(req),
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "stream": True,
                },
                stream=True,
            )

            # Hold back the sentence still being written. If the model stops on
            # its own it is flushed; if it hit max_tokens it is dropped, so the
            # user never sees a reply cut off mid-sentence.
            pending = ""
            hit_limit = False
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and line != "data: [DONE]":
                        choice = json.loads(line[6:])["choices"][0]
                        hit_limit = hit_limit or choice.get("finish_reason") == "length"
                        token = choice["delta"].get("content", "")
                        if token:
                            pending += token
                            ends = list(re.finditer(r'[.!?]["\')\]]?(?=\s)', pending))
                            if ends:
                                ready, pending = pending[: ends[-1].end()], pending[ends[-1].end():]
                                final_text += ready
                                yield f"data: {json.dumps({'token': ready})}\n\n"

            if hit_limit and final_text:
                rest = _trim_to_sentence(pending) if _SENTENCE_END.search(pending) else ""
            else:
                rest = pending
            if rest:
                final_text += rest
                yield f"data: {json.dumps({'token': rest})}\n\n"

        except Exception as e:
            error_obj = {"stage": "hf_endpoint", "message": str(e)}
            yield f"data: {json.dumps({'token': f'[Error: {e}]'})}\n\n"

        try:
            results = _retrieve_for_turn(req.message, final_text, k=int(req.top_k_sources))
            sources = format_sources(results, max_sources=int(req.top_k_sources))
            retrieval_struct = _structured_retrieval(results, max_sources=int(req.top_k_sources))

            if sources:
                yield f"data: {json.dumps({'token': sources})}\n\n"

        except Exception as e:
            error_obj = {"stage": "retrieval", "message": str(e)}
            yield f"data: {json.dumps({'token': f'[Warning: retrieval failed: {e}]'})}\n\n"

        try:
            combined = final_text + (sources or "")
            log_interaction_jsonl(
                session_id=x_session_id or "",
                endpoint="/generate_stream",
                mode="hf_endpoint",
                req=req,
                answer_text=combined,
                retrieval_struct=retrieval_struct,
                error=error_obj,
            )
        except Exception as e:
            print(f"[backend][log] failed: {e}")

        yield f"data: {json.dumps({'event': 'meta', 'interaction_id': interaction_id})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/comment")
def comment(
    req: CommentRequest,
    x_api_key: Optional[str] = Header(default=None),
    x_session_id: Optional[str] = Header(default=None, alias="sesh_id"),
):
    auth_check(x_api_key)

    if not x_session_id:
        raise HTTPException(status_code=400, detail="Missing sesh_id header")

    log_comment_jsonl(
        session_id=x_session_id,
        interaction_id=req.interaction_id,
        stars=req.stars,
        comment=req.comment or "",
    )
    return {"ok": True}

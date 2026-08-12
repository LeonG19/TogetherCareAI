import os
import json
from threading import Lock
from typing import Any, Dict, List, Tuple, Optional
from pinecone import Pinecone
import requests

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sentence_transformers import SentenceTransformer

from datetime import datetime
from pathlib import Path

import uuid

# ---------------------- CONFIG ----------------------
HF_TOKEN = os.environ.get("HF_TOKEN")

API_KEY = os.environ.get("API_KEY")  # optional

LOG_DIR = os.environ.get("LOG_DIR", "logs")
LOG_ENABLED = os.environ.get("LOG_ENABLED", "1") == "1"
_log_lock = Lock()

AB_ENABLED = os.environ.get("AB_ENABLED", "0") == "0"
AB_EVERY_N = int(os.environ.get("AB_EVERY_N", "5"))

COMMENTS_DIR = os.environ.get("COMMENTS_DIR", "comments")
COMMENTS_ENABLED = os.environ.get("COMMENTS_ENABLED", "1") == "1"

HF_ENDPOINT = "https://x2cc012krs19t4x5.eu-west-1.aws.endpoints.huggingface.cloud"

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "empathy-rag")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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


# ---------------------- LOAD MODELS ON IMPORT ----------------------
print("[backend] loading embedder + pinecone...")
embedder = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

print(f"[backend] ready. hf_endpoint={HF_ENDPOINT}, pinecone_index={PINECONE_INDEX_NAME}")


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


def embed_text(text: str) -> np.ndarray:
    vec = embedder.encode([text], normalize_embeddings=True, convert_to_numpy=True)
    return vec.astype(np.float32)


def retrieve_top_k(answer_text: str, k: int = 3) -> List[Tuple[float, Dict[str, Any]]]:
    q = embed_text(answer_text)[0].tolist()

    results = pinecone_index.query(
        vector=q,
        top_k=k,
        include_metadata=True,
    )

    out: List[Tuple[float, Dict[str, Any]]] = []

    for match in results.get("matches", []):
        score = float(match.get("score", 0.7))
        meta = match.get("metadata", {}) or {}
        out.append((score, meta))

    return out


def format_sources(results: List[Tuple[float, Dict[str, Any]]], max_sources: int = 3, min_score: float = 0.7) -> str:
    seen = set()
    lines = []
    added = 0
    for score, row in results:
        if score < min_score:
            continue
        url = extract_url(row)
        if not url or url in seen:
            continue
        seen.add(url)
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
    return row.get("id") or row.get("path") or "source"


def _structured_retrieval(results: List[Tuple[float, Dict[str, Any]]], max_sources: int) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for score, row in results:
        url = extract_url(row)
        if not url or url in seen:
            continue
        seen.add(url)
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


def _call_hf_endpoint(messages: List[Dict[str, str]], max_new_tokens: int, temperature: float, top_p: float) -> str:
    """Call the vLLM endpoint via OpenAI-compatible API and return the generated text."""
    response = requests.post(
        HF_ENDPOINT + "/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        json={
            "model": "ERivas8/empathy-llama3-8b",
            "messages": messages,
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        },
        timeout=300,
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


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
@app.get("/health")
def health():
    return {"ok": True, "hf_endpoint": HF_ENDPOINT, "pinecone_index": PINECONE_INDEX_NAME}


@app.post("/generate", response_model=GenerateResponse)
def generate(
    req: GenerateRequest,
    x_api_key: Optional[str] = Header(default=None),
    x_session_id: Optional[str] = Header(default=None, alias="sesh_id"),
):
    auth_check(x_api_key)

    messages = _truncate_history(list(req.history) if req.history else [])
    messages.append({"role": "user", "content": req.message})
    length_instruction = "Keep the response concise and do not exceed the allowed response length of 400 characters."
    messages = [{"role": "system", "content": length_instruction}] + messages

    try:
        answer = _call_hf_endpoint(messages, req.max_new_tokens, req.temperature, req.top_p)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"HF endpoint error: {e}")

    retrieval_struct = []
    sources = ""
    try:
        results = retrieve_top_k(answer, k=int(req.top_k_sources))
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

    style_a = "Be concise and direct. Keep the response concise and do not exceed the allowed response length of 400 characters."
    style_b = "Be warm and empathic. Keep the response concise and do not exceed the allowed response length of 400 characters."

    messages_a = [{"role": "system", "content": style_a}] + base_messages
    messages_b = [{"role": "system", "content": style_b}] + base_messages

    answer_a = _call_hf_endpoint(messages_a, req.max_new_tokens, req.temperature, req.top_p)
    answer_b = _call_hf_endpoint(messages_b, req.max_new_tokens, req.temperature, req.top_p)

    sources_a = ""
    sources_b = ""
    retr_a = []
    retr_b = []

    try:
        res_a = retrieve_top_k(answer_a, k=int(req.top_k_sources))
        sources_a = format_sources(res_a, max_sources=int(req.top_k_sources))
        retr_a = _structured_retrieval(res_a, max_sources=int(req.top_k_sources))
        if sources_a:
            answer_a += sources_a
    except Exception as e:
        print(f"[backend][ab][retrieval A] failed: {e}")

    try:
        res_b = retrieve_top_k(answer_b, k=int(req.top_k_sources))
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

    messages = _truncate_history(list(req.history) if req.history else [])
    messages.append({"role": "user", "content": req.message})
    length_instruction = "Keep the response concise and do not exceed the allowed response length of 400 characters."
    messages = [{"role": "system", "content": length_instruction}] + messages

    def event_generator():
        interaction_id = uuid.uuid4().hex

        final_text = ""
        sources = ""
        retrieval_struct = []
        error_obj = None

        try:
            response = requests.post(
                HF_ENDPOINT + "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "ERivas8/empathy-llama3-8b",
                    "messages": messages,
                    "max_tokens": req.max_new_tokens,
                    "temperature": req.temperature,
                    "top_p": req.top_p,
                    "stream": True,
                },
                stream=True,
                timeout=300,
            )

            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: ") and line != "data: [DONE]":
                        chunk = json.loads(line[6:])
                        token = chunk["choices"][0]["delta"].get("content", "")
                        if token:
                            final_text += token
                            yield f"data: {json.dumps({'token': token})}\n\n"

        except Exception as e:
            error_obj = {"stage": "hf_endpoint", "message": str(e)}
            yield f"data: {json.dumps({'token': f'[Error: {e}]'})}\n\n"

        try:
            results = retrieve_top_k(final_text, k=int(req.top_k_sources))
            print("[debug] pinecone results:", results[:3])
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

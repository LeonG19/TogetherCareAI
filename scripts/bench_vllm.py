#!/usr/bin/env python3
"""Measure time-to-first-token and tokens/sec against the vLLM server.

Baseline numbers to compare a local vLLM deployment against the hosted HF
Inference Endpoint. Uses only the stdlib so it runs in the slim backend venv.

  python scripts/bench_vllm.py                 # reads backend/.env
  python scripts/bench_vllm.py --tokens 200
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path


def env_from_file(p: Path) -> dict:
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument(
        "--prompt",
        default="Explain what sundowning is in dementia care, why it happens, "
                "and three practical things a family caregiver can try. "
                "Write about 200 words.",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    env = env_from_file(root / "backend" / ".env")
    url = env["HF_ENDPOINT"].rstrip("/") + "/v1/chat/completions"
    key = env["HF_TOKEN"]
    model = env["HF_MODEL"]

    print(f"endpoint : {url}")
    print(f"model    : {model}")
    print(f"target   : {args.tokens} tokens, {args.runs} runs\n")

    ttfts, rates, counts = [], [], []

    for run in range(1, args.runs + 1):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": args.prompt}],
            "max_tokens": args.tokens,
            "temperature": 0.7,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
        )

        t0 = time.perf_counter()
        ttft = None
        n_chunks = 0
        completion_tokens = None

        with urllib.request.urlopen(req) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                chunk = json.loads(body)
                usage = chunk.get("usage")
                if usage:
                    completion_tokens = usage.get("completion_tokens")
                choices = chunk.get("choices") or []
                if choices:
                    tok = choices[0].get("delta", {}).get("content") or ""
                    if tok:
                        if ttft is None:
                            ttft = time.perf_counter() - t0
                        n_chunks += 1
        total = time.perf_counter() - t0

        n = completion_tokens or n_chunks
        # tokens/sec after the first token: the decode rate proper
        decode = (n - 1) / (total - ttft) if ttft and total > ttft and n > 1 else 0.0
        ttfts.append(ttft or 0.0)
        rates.append(decode)
        counts.append(n)
        print(
            f"run {run}: ttft {ttft*1000:7.1f} ms | {n:4d} tokens | "
            f"total {total:6.2f}s | decode {decode:6.1f} tok/s"
        )

    n = len(ttfts)
    print(
        f"\nmean : ttft {sum(ttfts)/n*1000:.1f} ms | "
        f"{sum(counts)/n:.0f} tokens | {sum(rates)/n:.1f} tok/s"
    )


if __name__ == "__main__":
    main()

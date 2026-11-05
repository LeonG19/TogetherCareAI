#!/usr/bin/env python
"""Merge a PEFT/LoRA adapter into Llama-3.1-8B-Instruct and export a standalone
model directory that vLLM can serve directly.

The adapters in llm-healthcare/ML_2025 were trained with 4-bit QLoRA, but their
own ADAPTER_METADATA.json says to load the base in (b)float16 with no
quantisation when merging at deploy time -- so that is what this does.

Runs entirely on CPU: the GPUs on this box belong to other people, and a
merge is a one-shot matmul pass that does not need one.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="path to base model snapshot dir")
    ap.add_argument("--adapter", required=True, help="path to LoRA adapter dir")
    ap.add_argument("--out", required=True, help="output dir for the merged model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        print(f"[merge] refusing to overwrite non-empty {out}", file=sys.stderr)
        return 1

    print(f"[merge] base    : {args.base}")
    print(f"[merge] adapter : {args.adapter}")
    print(f"[merge] dtype   : {args.dtype}")

    print("[merge] loading base on CPU (this reads ~16 GB)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )

    print("[merge] applying adapter...")
    model = PeftModel.from_pretrained(model, args.adapter, dtype=dtype)

    print("[merge] merge_and_unload()...")
    model = model.merge_and_unload()
    model = model.to(dtype)

    print(f"[merge] writing merged model to {out} ...")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)

    # Copy the BASE tokenizer files verbatim rather than round-tripping them
    # through AutoTokenizer.save_pretrained(). Two reasons, both learned the
    # hard way:
    #
    #  1. Format drift. save_pretrained() writes the tokenizer in the format of
    #     whatever transformers version is doing the merging. transformers 5.x
    #     emits {"tokenizer_class": "TokenizersBackend"}, which a transformers
    #     4.x runtime (what vLLM 0.11.x pins) cannot resolve -- the server dies
    #     at load with "Tokenizer class TokenizersBackend does not exist".
    #  2. Chat template placement. transformers 5.x splits chat_template out of
    #     tokenizer_config.json into a sibling chat_template.jinja. Copying the
    #     base files keeps the template inline exactly where the base had it,
    #     so vLLM's /v1/chat/completions route finds it with no --chat-template
    #     flag. The adapter dirs cannot supply this: they ship only the .jinja
    #     sidecar and have no chat_template key at all.
    print("[merge] copying tokenizer files from base (verbatim, no re-save)...")
    base = Path(args.base)
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
    ):
        src_f = base / name
        if src_f.exists():
            shutil.copy2(src_f, out / name)
            print(f"[merge]   {name}")

    # Drop any sidecar the model save may have produced, so the chat template
    # has exactly one source of truth (inline in tokenizer_config.json).
    sidecar = out / "chat_template.jinja"
    if sidecar.exists():
        sidecar.unlink()

    tok_cfg = json.loads((out / "tokenizer_config.json").read_text())
    if not isinstance(tok_cfg.get("chat_template"), str):
        print("[merge] WARNING: no inline chat_template -- vLLM's chat route "
              "will need --chat-template", file=sys.stderr)
    else:
        print(f"[merge] chat_template inline ({len(tok_cfg['chat_template'])} chars), "
              f"tokenizer_class={tok_cfg.get('tokenizer_class')}")

    # Carry the training-time prompt across if the adapter recorded one.
    for extra in ("ADAPTER_METADATA.json", "DPO_PROMPT.json"):
        src = Path(args.adapter) / extra
        if src.exists():
            shutil.copy2(src, out / extra)
            print(f"[merge] copied {extra}")

    cfg = json.loads((out / "config.json").read_text())
    print(f"[merge] done. dtype={cfg.get('torch_dtype') or cfg.get('dtype')} "
          f"vocab={cfg.get('vocab_size')} layers={cfg.get('num_hidden_layers')}")
    total = sum(f.stat().st_size for f in out.glob("*.safetensors"))
    print(f"[merge] weights on disk: {total / 1e9:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

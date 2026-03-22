"""
Export a Hugging Face tokenizer to tokenizer.bin

Binary format: u32 max_token_length, then for each id 0..vocab_size-1:
  f32 score (unused by qwen35 encode; written as 0), u32 byte length, utf-8 bytes

Row count matches config text vocab_size so it aligns with export.py / the C checkpoint
"""

import argparse
import json
import os
import struct

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import AutoTokenizer


def find_cached_models(cache_dir: str) -> list[str]:
    """Return snapshot dirs for Qwen3.5 models found in cache_dir."""
    found = []
    if not os.path.isdir(cache_dir):
        return found
    for model_name in os.listdir(cache_dir):
        if "Qwen3.5" not in model_name:
            continue
        snapshots_dir = os.path.join(cache_dir, model_name, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue
        for rev in os.listdir(snapshots_dir):
            snap = os.path.join(snapshots_dir, rev)
            if os.path.isfile(os.path.join(snap, "config.json")) and (
                os.path.isfile(os.path.join(snap, "tokenizer.json"))
                or os.path.isfile(os.path.join(snap, "tokenizer_config.json"))
            ):
                found.append(snap)
    return found


def resolve_model_dir(source: str, cache_dir: str) -> str:
    if os.path.isdir(source):
        return os.path.abspath(source)
    if "/" in source and not os.path.exists(source):
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=source, cache_dir=cache_dir)
    return os.path.abspath(os.path.dirname(source))


def load_vocab_size(config_path: str) -> int:
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    llm = config.get("text_config", config)
    return int(llm["vocab_size"])


def id_to_utf8_bytes(tokenizer, idx: int) -> bytes:
    piece = tokenizer.convert_ids_to_tokens(idx)
    if piece is None:
        return b""
    content = getattr(piece, "content", piece)
    if not isinstance(content, str):
        content = str(content)
    s = tokenizer.convert_tokens_to_string([content])
    return s.encode("utf-8")


def export(model_dir: str, output: str) -> None:
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found under {model_dir}")

    vocab_size = load_vocab_size(config_path)
    print(f"config vocab_size: {vocab_size}")

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)

    tv = getattr(tok, "vocab_size", None)
    if tv is not None and tv != vocab_size:
        print(
            f"note: tokenizer.vocab_size={tv} != config vocab_size; "
            f"exporting {vocab_size} rows to match checkpoint"
        )

    tokens: list[bytes] = []
    scores: list[float] = []
    for i in range(vocab_size):
        tokens.append(id_to_utf8_bytes(tok, i))
        scores.append(0.0)

    max_token_length = max((len(t) for t in tokens), default=0)
    print(f"max_token_length: {max_token_length}")

    with open(output, "wb") as f:
        f.write(struct.pack("I", max_token_length))
        for score, bs in zip(scores, tokens):
            f.write(struct.pack("fI", score, len(bs)))
            f.write(bs)

    print(f"wrote {output} ({vocab_size} tokens)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HF tokenizer to tokenizer.bin for qwen35.c"
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=str,
        help=(
            "HF repo id or local model directory. "
            "If omitted, auto-detected from the local cache."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="tokenizer.bin",
        help="Output path (default: tokenizer.bin)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=(
            "HuggingFace cache directory "
            f"(default: {HUGGINGFACE_HUB_CACHE})"
        ),
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir or HUGGINGFACE_HUB_CACHE

    if args.source:
        model_dir = resolve_model_dir(args.source, cache_dir)
    else:
        candidates = find_cached_models(cache_dir)
        if not candidates:
            print(
                f"No Qwen3.5 model found in cache ({cache_dir}).\n"
                "Download a model first, e.g.:\n"
                "  python export.py Qwen/Qwen3.5-0.8B"
            )
            raise SystemExit(1)

        if len(candidates) == 1:
            model_dir = candidates[0]
            print(f"Found model: {model_dir}")
        else:
            print("Multiple models found in cache:")
            for i, c in enumerate(candidates):
                print(f"  [{i}] {c}")
            choice = input("Select model [0]: ").strip()
            idx = int(choice) if choice else 0
            model_dir = candidates[idx]

    export(model_dir, args.output)


if __name__ == "__main__":
    main()

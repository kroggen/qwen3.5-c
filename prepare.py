"""
Prepare Qwen-3.5 model for inference in C.
Downloads model from HuggingFace, creates tokenizer.bin, and saves config.
"""

import argparse
import json
import os
import struct

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
from transformers import AutoTokenizer

CONFIG_FILE = "models.json"


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def save_config(config: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def download_model(model_name: str, cache_dir: str = None) -> str:
    print(f"Downloading model: {model_name}")
    model_path = snapshot_download(repo_id=model_name, cache_dir=cache_dir)
    print(f"Model downloaded to: {model_path}")
    return model_path


def create_tokenizer(model_dir: str, output: str = None) -> str:
    if output is None:
        output = os.path.join(model_dir, "tokenizer.bin")

    if os.path.exists(output):
        print(f"Tokenizer already exists: {output}")
        return output

    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found under {model_dir}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    llm = config.get("text_config", config)
    vocab_size = int(llm["vocab_size"])
    print(f"Config vocab_size: {vocab_size}")

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)

    tv = getattr(tok, "vocab_size", None)
    if tv is not None and tv != vocab_size:
        print(f"Note: tokenizer.vocab_size={tv} != config vocab_size; "
              f"exporting {vocab_size} rows to match checkpoint")

    def id_to_utf8_bytes(idx: int) -> bytes:
        piece = tok.convert_ids_to_tokens(idx)
        if piece is None:
            return b""
        content = getattr(piece, "content", piece)
        if not isinstance(content, str):
            content = str(content)
        s = tok.convert_tokens_to_string([content])
        return s.encode("utf-8")

    tokens = []
    for i in range(vocab_size):
        tokens.append(id_to_utf8_bytes(i))

    max_token_length = max((len(t) for t in tokens), default=0)
    print(f"Max token length: {max_token_length}")

    with open(output, "wb") as f:
        f.write(struct.pack("I", max_token_length))
        for bs in tokens:
            f.write(struct.pack("fI", 0.0, len(bs)))
            f.write(bs)

    print(f"Created tokenizer: {output} ({vocab_size} tokens)")
    return output


def prepare(model_name: str, cache_dir: str = None) -> str:
    model_path = download_model(model_name, cache_dir)

    tokenizer_path = create_tokenizer(model_path)

    config = load_config()
    config[model_name] = {
        "path": model_path,
        "tokenizer": tokenizer_path
    }
    save_config(config)

    print(f"\nModel prepared successfully!")
    print(f"Model name: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Tokenizer:  {tokenizer_path}")
    print(f"\nRun with: ./qwen35 {model_name}")

    return model_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Qwen-3.5 model for C inference"
    )
    parser.add_argument(
        "model",
        type=str,
        help="HuggingFace model name (e.g., Qwen/Qwen3.5-0.8B)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help=f"HuggingFace cache directory (default: {HUGGINGFACE_HUB_CACHE})"
    )
    args = parser.parse_args()

    prepare(args.model, args.cache_dir)


if __name__ == "__main__":
    main()

"""
Export Qwen-3.5 model weights to a binary format for inference in C.
Supports both full attention and linear attention (GatedDeltaNet) layers.
Weights are grouped by type for efficient memory mapping.
"""

import os
import struct
import argparse
import json

import numpy as np
import torch

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

QWEN35_MAGIC = 0x51773335
VERSION = 1


def serialize_fp32(file, tensor):
    t = tensor.detach().cpu().contiguous().view(-1).to(torch.float32)
    file.write(t.numpy().tobytes())


def serialize_int32(file, val):
    file.write(struct.pack("i", val))


def serialize_float32(file, val):
    file.write(struct.pack("f", val))


def load_model(path, cache_dir=None):
    print(f"loading model from {path}")

    if "/" in path and not os.path.exists(path):
        print(f"Downloading from HuggingFace: {path}")
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=path, cache_dir=cache_dir)
        print(f"Downloaded to: {path}")

    if os.path.isdir(path):
        safetensors_path = os.path.join(path, "model.safetensors")
        safetensors_pattern = os.path.join(path, "model.safetensors-*.safetensors")
        pytorch_path = os.path.join(path, "pytorch_model.bin")
        config_path = os.path.join(path, "config.json")

        if os.path.exists(safetensors_path):
            from safetensors.torch import load_file
            model = load_file(safetensors_path)
        else:
            import glob
            safetensors_files = glob.glob(safetensors_pattern)
            if safetensors_files:
                from safetensors.torch import load_file
                if len(safetensors_files) == 1:
                    model = load_file(safetensors_files[0])
                else:
                    model = {}
                    for f in sorted(safetensors_files):
                        shard = load_file(f)
                        model.update(shard)
            elif os.path.exists(pytorch_path):
                model = torch.load(pytorch_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"No model file found in {path}")
    else:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            model = load_file(path)
        else:
            model = torch.load(path, map_location="cpu")
        config_path = os.path.join(os.path.dirname(path), "config.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    return model, config


def get_layer_type(layer_idx, layer_types):
    if layer_types is None:
        return "full_attention"
    return layer_types[layer_idx] if layer_idx < len(layer_types) else "full_attention"


def export_qwen35(model_weights, config, filepath, max_seq_len=2048):
    """
    Export Qwen-3.5 model weights to binary format.

    Layout (grouped by weight type for efficient memory mapping):
    - Header (256 bytes)
    - token_embedding_table
    - rms_att_weight (all layers input layernorm)
    - Full attention weights (for attention layers only, indexed by 'la'):
      - wq (all attention layers)
      - wk (all attention layers)
      - wv (all attention layers)
      - wo (all attention layers)
      - q_norm (all attention layers)
      - k_norm (all attention layers)
    - Linear attention weights (for deltanet layers only, indexed by 'ld'):
      - in_proj_qkv (all deltanet layers)
      - in_proj_z (all deltanet layers)
      - in_proj_b (all deltanet layers)
      - in_proj_a (all deltanet layers)
      - conv1d_weight (all deltanet layers)
      - dt_bias (all deltanet layers)
      - A_log (all deltanet layers)
      - linear_norm (all deltanet layers)
      - out_proj (all deltanet layers)
    - rms_ffn_weight (all layers post layernorm)
    - w1 (all layers)
    - w2 (all layers)
    - w3 (all layers)
    - rms_final_weight
    - (optional) wcls if not tied
    """

    llm_config = config.get("text_config", config)

    dim = llm_config["hidden_size"]
    n_heads = llm_config["num_attention_heads"]
    n_kv_heads = llm_config.get("num_key_value_heads", n_heads)
    n_layer = llm_config["num_hidden_layers"]

    n_mlp = llm_config.get("intermediate_size")
    if n_mlp is None:
        n_mlp = llm_config.get("shared_expert_intermediate_size")
    if n_mlp is None:
        n_mlp = llm_config.get("moe_intermediate_size")

    vocab_size = llm_config["vocab_size"]
    rope_theta = llm_config.get("rope_theta", 10000.0)
    rms_norm_eps = llm_config.get("rms_norm_eps", 1e-6)

    d_head = llm_config.get("head_dim", dim // n_heads)

    layer_types = llm_config.get("layer_types")
    n_linear_k_heads = llm_config.get("linear_num_key_heads", 0)
    n_linear_v_heads = llm_config.get("linear_num_value_heads", 0)
    d_linear_k = llm_config.get("linear_key_head_dim", 0)
    d_linear_v = llm_config.get("linear_value_head_dim", 0)
    linear_conv_kernel = llm_config.get("linear_conv_kernel_dim", 4)

    tie_word_embeddings = config.get("tie_word_embeddings", False)

    attention_layers = []
    deltanet_layers = []
    layer_type_flags = []
    for i in range(n_layer):
        lt = get_layer_type(i, layer_types)
        if lt == "linear_attention":
            layer_type_flags.append(1)
            deltanet_layers.append(i)
        else:
            layer_type_flags.append(0)
            attention_layers.append(i)

    n_full_attn = len(attention_layers)
    n_linear_attn = len(deltanet_layers)

    print(f"Model config:")
    print(f"  dim: {dim}")
    print(f"  n_heads: {n_heads}")
    print(f"  n_kv_heads: {n_kv_heads}")
    print(f"  n_layer: {n_layer}")
    print(f"  n_mlp: {n_mlp}")
    print(f"  vocab_size: {vocab_size}")
    print(f"  d_head: {d_head}")
    print(f"  rope_theta: {rope_theta}")
    print(f"  rms_norm_eps: {rms_norm_eps}")
    print(f"  tie_word_embeddings: {tie_word_embeddings}")
    print(f"  n_full_attn_layers: {n_full_attn}")
    print(f"  n_linear_attn_layers: {n_linear_attn}")
    print(f"  attention_layers: {attention_layers}")
    print(f"  deltanet_layers: {deltanet_layers}")
    if n_linear_attn > 0:
        print(f"  n_linear_k_heads: {n_linear_k_heads}")
        print(f"  n_linear_v_heads: {n_linear_v_heads}")
        print(f"  d_linear_k: {d_linear_k}")
        print(f"  d_linear_v: {d_linear_v}")
        print(f"  linear_conv_kernel: {linear_conv_kernel}")

    out_file = open(filepath, "wb")

    out_file.write(struct.pack("I", QWEN35_MAGIC))
    out_file.write(struct.pack("i", VERSION))

    out_file.write(struct.pack("i", dim))
    out_file.write(struct.pack("i", n_heads))
    out_file.write(struct.pack("i", n_kv_heads))
    out_file.write(struct.pack("i", n_layer))
    out_file.write(struct.pack("i", n_mlp))
    out_file.write(struct.pack("i", vocab_size))
    out_file.write(struct.pack("i", max_seq_len))
    out_file.write(struct.pack("f", rope_theta))
    out_file.write(struct.pack("f", rms_norm_eps))
    out_file.write(struct.pack("i", 1 if tie_word_embeddings else 0))
    out_file.write(struct.pack("i", d_head))
    out_file.write(struct.pack("i", n_linear_k_heads))
    out_file.write(struct.pack("i", n_linear_v_heads))
    out_file.write(struct.pack("i", d_linear_k))
    out_file.write(struct.pack("i", d_linear_v))
    out_file.write(struct.pack("i", linear_conv_kernel))
    out_file.write(struct.pack("i", n_full_attn))
    out_file.write(struct.pack("i", n_linear_attn))

    for lt in layer_type_flags:
        out_file.write(struct.pack("i", lt))

    pad = 256 - out_file.tell()
    assert pad >= 0, f"Header overflow! Need {256 - pad} bytes but header is 256"
    out_file.write(b"\0" * pad)

    def get_weight(key):
        if key in model_weights:
            return model_weights[key]
        for k, v in model_weights.items():
            if k == key or k.endswith("." + key):
                return v
        parts = key.split(".")
        for i in range(len(parts)):
            alt_key = "model.language_model." + ".".join(parts[i:])
            if alt_key in model_weights:
                return model_weights[alt_key]
        return None

    embed_key = "model.embed_tokens.weight"
    embed = get_weight(embed_key)
    if embed is None:
        embed_key = "model.language_model.embed_tokens.weight"
        embed = get_weight(embed_key)
    if embed is None:
        for k in model_weights:
            if "embed_tokens" in k:
                embed = model_weights[k]
                embed_key = k
                break
    if embed is None:
        raise ValueError("Could not find embedding weights")
    print(f"writing {embed_key} {list(embed.shape)[::-1]}")
    serialize_fp32(out_file, embed)

    print(f"writing input_layernorm for all {n_layer} layers")
    for l in range(n_layer):
        layer_prefix = f"model.layers.{l}"
        norm_key = f"{layer_prefix}.input_layernorm.weight"
        w = get_weight(norm_key)
        if w is None:
            norm_key = f"model.language_model.layers.{l}.input_layernorm.weight"
            w = get_weight(norm_key)
        if w is None:
            raise ValueError(f"Could not find input_layernorm for layer {l}")
        serialize_fp32(out_file, w)

    kv_dim = n_kv_heads * d_head
    key_dim = n_linear_k_heads * d_linear_k
    value_dim = n_linear_v_heads * d_linear_v
    conv_dim = key_dim * 2 + value_dim

    print(f"writing wq for {n_full_attn} attention layers")
    for l in attention_layers:
        layer_prefix = f"model.layers.{l}"
        wq_key = f"{layer_prefix}.self_attn.q_proj.weight"
        w = get_weight(wq_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: wq not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(n_heads * d_head * 2, dim))

    print(f"writing wk for {n_full_attn} attention layers")
    for l in attention_layers:
        layer_prefix = f"model.layers.{l}"
        wk_key = f"{layer_prefix}.self_attn.k_proj.weight"
        w = get_weight(wk_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: wk not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(kv_dim, dim))

    print(f"writing wv for {n_full_attn} attention layers")
    for l in attention_layers:
        layer_prefix = f"model.layers.{l}"
        wv_key = f"{layer_prefix}.self_attn.v_proj.weight"
        w = get_weight(wv_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: wv not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(kv_dim, dim))

    print(f"writing wo for {n_full_attn} attention layers")
    for l in attention_layers:
        layer_prefix = f"model.layers.{l}"
        wo_key = f"{layer_prefix}.self_attn.o_proj.weight"
        w = get_weight(wo_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: wo not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(dim, n_heads * d_head))

    print(f"writing q_norm for {n_full_attn} attention layers")
    for l in attention_layers:
        layer_prefix = f"model.layers.{l}"
        q_norm_key = f"{layer_prefix}.self_attn.q_norm.weight"
        w = get_weight(q_norm_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: q_norm not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(d_head))

    print(f"writing k_norm for {n_full_attn} attention layers")
    for l in attention_layers:
        layer_prefix = f"model.layers.{l}"
        k_norm_key = f"{layer_prefix}.self_attn.k_norm.weight"
        w = get_weight(k_norm_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: k_norm not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(d_head))

    print(f"writing in_proj_qkv for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.in_proj_qkv.weight"
        w = get_weight(key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: in_proj_qkv not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(conv_dim, dim))

    print(f"writing in_proj_z for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.in_proj_z.weight"
        w = get_weight(key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: in_proj_z not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(value_dim, dim))

    print(f"writing in_proj_b for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.in_proj_b.weight"
        w = get_weight(key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: in_proj_b not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(n_linear_v_heads, dim))

    print(f"writing in_proj_a for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.in_proj_a.weight"
        w = get_weight(key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: in_proj_a not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(n_linear_v_heads, dim))

    print(f"writing conv1d_weight for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.conv1d.weight"
        w = get_weight(key)
        if w is not None:
            if len(w.shape) == 3:
                w = w.squeeze(1)
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: conv1d not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(conv_dim, linear_conv_kernel))

    print(f"writing dt_bias for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.dt_bias"
        w = get_weight(key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: dt_bias not found for layer {l}")
            serialize_fp32(out_file, torch.ones(n_linear_v_heads))

    print(f"writing A_log for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.A_log"
        w = get_weight(key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: A_log not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(n_linear_v_heads))

    print(f"writing linear_norm for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.norm.weight"
        w = get_weight(key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: linear norm not found for layer {l}")
            serialize_fp32(out_file, torch.ones(value_dim))

    print(f"writing out_proj for {n_linear_attn} deltanet layers")
    for l in deltanet_layers:
        layer_prefix = f"model.layers.{l}"
        key = f"{layer_prefix}.linear_attn.out_proj.weight"
        w = get_weight(key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: out_proj not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(dim, value_dim))

    print(f"writing post_attention_layernorm for all {n_layer} layers")
    for l in range(n_layer):
        layer_prefix = f"model.layers.{l}"
        norm_key = f"{layer_prefix}.post_attention_layernorm.weight"
        w = get_weight(norm_key)
        if w is None:
            norm_key = f"model.language_model.layers.{l}.post_attention_layernorm.weight"
            w = get_weight(norm_key)
        if w is None:
            raise ValueError(f"Could not find post_attention_layernorm for layer {l}")
        serialize_fp32(out_file, w)

    print(f"writing w1 (gate_proj) for all {n_layer} layers")
    for l in range(n_layer):
        layer_prefix = f"model.layers.{l}"
        w1_key = f"{layer_prefix}.mlp.gate_proj.weight"
        w = get_weight(w1_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: w1 not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(n_mlp, dim))

    print(f"writing w2 (down_proj) for all {n_layer} layers")
    for l in range(n_layer):
        layer_prefix = f"model.layers.{l}"
        w2_key = f"{layer_prefix}.mlp.down_proj.weight"
        w = get_weight(w2_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: w2 not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(dim, n_mlp))

    print(f"writing w3 (up_proj) for all {n_layer} layers")
    for l in range(n_layer):
        layer_prefix = f"model.layers.{l}"
        w3_key = f"{layer_prefix}.mlp.up_proj.weight"
        w = get_weight(w3_key)
        if w is not None:
            serialize_fp32(out_file, w)
        else:
            print(f"Warning: w3 not found for layer {l}")
            serialize_fp32(out_file, torch.zeros(n_mlp, dim))

    final_norm_key = "model.norm.weight"
    w = get_weight(final_norm_key)
    if w is None:
        final_norm_key = "model.language_model.norm.weight"
        w = get_weight(final_norm_key)
    if w is None:
        for k in model_weights:
            if "norm.weight" in k and "layernorm" not in k.lower() and "layers" not in k:
                w = model_weights[k]
                final_norm_key = k
                break
    if w is None:
        raise ValueError("Could not find final norm weights")
    print(f"writing final norm {list(w.shape)}")
    serialize_fp32(out_file, w)

    if not tie_word_embeddings:
        lm_head_key = "lm_head.weight"
        w = get_weight(lm_head_key)
        if w is not None:
            print(f"writing lm_head {list(w.shape)[::-1]}")
            serialize_fp32(out_file, w)
        else:
            print("Warning: lm_head not found, using embedding weights")

    out_file.close()
    print(f"\nExport complete: {filepath}")
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size / (1024 * 1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen-3.5 model to C-compatible binary format"
    )
    parser.add_argument(
        "source", type=str, help="Path to model directory or safetensors file"
    )
    parser.add_argument("destination", type=str, help="Output binary file path")
    parser.add_argument(
        "--max-seq-len", type=int, default=2048, help="Maximum sequence length"
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

    model_weights, config = load_model(args.source, cache_dir=args.cache_dir)
    export_qwen35(model_weights, config, args.destination, args.max_seq_len)


if __name__ == "__main__":
    main()

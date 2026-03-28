# Qwen3.5 in C

Inference of Qwen3.5 models in pure C, for learning purpose

No pytorch required. The safetensors loading is done with [safetensors-cpp](https://github.com/syoyo/safetensors-cpp)

Inspired by [llama2.c](https://github.com/karpathy/llama2.c) and [mamba.c](https://github.com/kroggen/mamba.c)

For those interested in (or that only learn by) seeing the actual operations on the weights and state at a lower level

Qwen 3.5 combines multi-head attention and linear attention (GatedDeltaNet) layers

For fast inference, use other methods like [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton)


## Quick Start

```bash
pip install huggingface_hub transformers
python prepare.py Qwen/Qwen3.5-0.8B   # download + create tokenizer
make fast
./qwen35 Qwen3.5-0.8B
```

If there are more than 1 model with the same name on the local cache, pass the full name of the model:
```
./qwen35 Qwen/Qwen3.5-0.8B
```

Or pass the path to the folder containing the model:
```bash
./qwen35 ./Qwen3.5-0.8B
```


## Models

Use Qwen3.5 **dense** models from [Qwen's Huggingface folder](https://huggingface.co/collections/Qwen/qwen35) or finetunes

Examples:

* Qwen/Qwen3.5-0.8B
* Qwen/Qwen3.5-2B
* Qwen/Qwen3.5-4B
* Qwen/Qwen3.5-9B

Many of these repos are vision–language. This implementation currently uses the text transformer only

Not supported: MoE models (`Qwen3.5-35B-A3B`, `122B-A10B`, `397B-A17B`...), FP8 / GPTQ / other quantized weight formats


## Build

```bash
make          # reference
make fast     # -Ofast -march=native (native SIMD when available)
make omp      # OpenMP (set OMP_NUM_THREADS when running)
make debug
make clean
```


## Run

```bash
./qwen35 <model> [options]

# <model> can be a model name (after prepare.py) or a local directory
./qwen35 Qwen3.5-0.8B
./qwen35 Qwen/Qwen3.5-2B -y "" -i "Hello!"
./qwen35 ./Qwen3.5-4B -y "You are a helpful assistant"
```


# License

WTFPL (Do What The Fuck You Want To Public License)

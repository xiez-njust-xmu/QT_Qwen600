English | [中文](README.md)

[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Qt](https://img.shields.io/badge/Qt-5.15-41CD52?style=flat-square&logo=qt)](https://www.qt.io/)
[![License](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)
[![Model](https://img.shields.io/badge/Model-Qwen3--0.6B-purple?style=flat-square)](https://huggingface.co/Qwen/Qwen3-0.6B)

<p align="center">
  <img src="assets/banner.png" width="600" alt="qwen600.qt banner">
</p>

# qwen600.qt

A desktop LLM inference application built on [qwen600.cu](https://github.com/yassa9/qwen600). Extends the original CUDA inference engine with a Qt GUI, KV Cache INT8 quantization, softmax kernel fix, and RoPE pre-computation optimization.

## Highlights

- **Qt Desktop Chat Interface** — Real-time streaming output, background inference thread, adjustable parameter panel
- **KV Cache INT8 Quantization** — Per-head symmetric quantization, ~50% KV cache VRAM reduction (3.5 GB → 1.8 GB)
- **Softmax Kernel Fix** — Adaptive multi-block-size dispatch with numerically stable parallel reduction
- **RoPE Pre-computed Tables** — One-time sin/cos table initialization, eliminating redundant trig calls during inference

## Screenshots

<p align="center">
  <img src="assets/screenshot.png" width="700" alt="qwen600.qt chat interface">
</p>

## Quick Start

### Requirements

| Dependency | Version |
|-----------|---------|
| CUDA Toolkit (nvcc, cuBLAS, CUB) | 12.0+ |
| Qt5 (Widgets module) | 5.15+ |
| CMake | 3.20+ |
| GPU VRAM | Recommended ≥ 6 GB |

### Model Download

Download [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) from Hugging Face. Ensure the directory contains `model.safetensors`.

```bash
# Verify weight file integrity
sha256sum model.safetensors
# Expected: f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b
```

Export the tokenizer:

```bash
python export.py <model_dir>
```

### Build

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### Configuration

Edit `app_config.h` to set the model path:

```cpp
#define MODEL_DIR "path/to/Qwen3-0.6B"
```

Edit `config.h` to toggle INT8 quantization:

```cpp
constexpr bool USE_KV_CACHE_INT8 = true;  // false for original BF16
```

### Run

Launch the application after building — the Qt window loads the model and enters chat mode:

```bash
./QT_Qwen600
```

## Technical Details

### KV Cache INT8 Quantization

**Problem:** BF16 KV cache consumes ~3.3 GB VRAM at 32K context length.

**Solution:** After QK-Norm and RoPE, quantize each token's K/V vectors at per-head granularity using symmetric INT8:

```
scale = max(|head_vector|) / 127
int8_value = clamp(round(value / scale), -127, 127)
```

During attention, dequantize on-the-fly: `float_value = int8_value × scale`

**Memory Comparison:**

| Component | BF16 | INT8 |
|-----------|------|------|
| K cache   | 1.75 GB | 875 MB |
| V cache   | 1.75 GB | 875 MB |
| Scale arrays | — | 57.4 MB |
| **Total** | **3.5 GB** | **~1.8 GB** |

**New CUDA Kernels:**
- `quantize_kv_head_kernel` — Per-head absmax quantization via `cub::BlockReduce`
- `attention_qk_kernel_int8` — QK dot product with on-the-fly INT8 K cache dequantization
- `attention_v_kernel_int8` — Weighted V sum with on-the-fly INT8 V cache dequantization

**Design Decisions:**
- RoPE and QK-Norm execute at full BF16 precision before quantization — no rotary encoding precision loss
- Scales stored as float32 to avoid additional precision degradation
- `if constexpr` compile-time branching — zero overhead for the inactive path

### Softmax Kernel Optimization

Implemented a numerically stable parallel softmax using `cub::BlockReduce` for max-reduce and sum-reduce. The dispatcher automatically selects the optimal block size (64 / 256 / 512 / 1024 threads) based on sequence length, preventing thread waste on short sequences or insufficient parallelism on long ones.

### RoPE Pre-computed Tables

`rope_precompute_init` computes all positional cos/sin values at model initialization, stored as a GPU-side lookup table (shape: `SEQ_LEN × HEAD_DIM/2`). During inference, `rope_gpu_table` indexes by position, replacing the original per-token `sincosf` / `powf` online computation.

### Qt GUI

A native desktop chat application built with Qt5 Widgets:

- **Async Inference:** `InferenceWorker` runs in a dedicated QThread, streaming tokens to the UI via signal-slot
- **Parameter Panel:** Real-time adjustment of temperature, top-p, top-k, and system prompt
- **Controls:** Send, stop generation, clear context
- **Dark Theme:** Catppuccin-inspired color scheme with monospace font rendering

## Project Structure

```
qwen600.qt/
├── CMakeLists.txt              Build config (CUDA + Qt5)
├── config.h                    Model constants & INT8 quantization toggle
├── app_config.h                Runtime config (model path, sampling params)
├── qwen_model.cuh              CUDA kernels (attention, quantization, RoPE, RMSNorm, SwiGLU)
├── static_loader.h             Safetensors weight loading (mmap + async copy)
├── tokenizer.h                 BPE tokenizer
├── sampler.h                   Top-k / Top-p / Temperature sampling
├── inference_engine.h          C++ inference API (PIMPL pattern)
├── inference_engine_impl.cu    Inference engine CUDA implementation
├── mainwindow.h / .cpp         Qt chat window
├── main.cpp                    Application entry point
└── assets/                     Image resources
```

## Acknowledgments

Built upon:

- [qwen600.cu](https://github.com/yassa9/qwen600) — Original inference engine by yassa9
- [llama2.c](https://github.com/karpathy/llama2.c) — Andrej Karpathy
- [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) — Qwen Team

## License

[MIT](LICENSE)
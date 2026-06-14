# qwen600.qt

A desktop chat application built on top of [qwen600.cu](https://github.com/yassa9/qwen600), extending the original CUDA inference engine with a Qt GUI, KV Cache INT8 quantization, and kernel-level optimizations.

## What This Project Adds

Based on the original `qwen600.cu` (a minimal BF16 CUDA inference engine for Qwen3-0.6B), this fork introduces:

| Feature | Description |
|---------|-------------|
| Qt Desktop GUI | Real-time streaming chat interface with parameter controls |
| KV Cache INT8 Quantization | ~50% KV cache VRAM reduction via per-head symmetric quantization |
| Softmax Kernel Fix | Corrected softmax dispatcher with proper multi-block-size selection |
| RoPE Pre-computed Tables | Pre-computed sin/cos lookup tables replacing on-the-fly RoPE calculation |

## Qt Chat Interface

A native desktop chat window replacing the original CLI, featuring:

- Real-time token-by-token streaming display
- Background inference thread (non-blocking UI)
- Adjustable generation parameters: temperature, top-p, top-k
- Configurable system prompt
- Stop generation / clear context controls
- Dark theme styled with Catppuccin-inspired colors

**Files added:** `mainwindow.h`, `mainwindow.cpp`, `main.cpp`, `app_config.h`, `inference_engine.h`, `inference_engine_impl.cu`

## KV Cache INT8 Quantization

### Problem

KV cache in BF16 consumes ~3.5 GB for full 32K context — nearly half the VRAM budget on an 8 GB GPU.

### Solution

Per-token per-head symmetric INT8 quantization with on-the-fly dequantization during attention.

- After QK-Norm + RoPE (computed at full BF16 precision), K/V vectors are quantized: `scale = max(|head|) / 127`
- Attention kernels dequantize INT8 cache entries during QK dot product and V aggregation
- Controlled by compile-time switch: `constexpr bool USE_KV_CACHE_INT8` in `config.h`

### Memory Comparison

| Component | BF16 (original) | INT8 (this work) |
|-----------|-----------------|------------------|
| K cache   | 1.75 GB         | 875 MB           |
| V cache   | 1.75 GB         | 875 MB           |
| Scale arrays | -            | 57.4 MB          |
| **Total** | **3.5 GB**      | **~1.8 GB**      |

### New CUDA Kernels

- `quantize_kv_head_kernel` — Per-head absmax INT8 quantization via `cub::BlockReduce`
- `attention_qk_kernel_int8` — QK dot product with on-the-fly dequantization
- `attention_v_kernel_int8` — Weighted V sum with on-the-fly dequantization

## Kernel Optimizations

### Softmax Kernel & Dispatcher

Fixed the softmax implementation with a proper multi-block-size dispatcher that selects optimal block size (64/256/512/1024 threads) based on sequence length, using `cub::BlockReduce` for numerically stable parallel max-reduce and sum-reduce.

### RoPE Pre-computed Tables

Replaced on-the-fly RoPE calculation with pre-computed cos/sin lookup tables (`rope_precompute_init`). The table is computed once at model initialization and indexed by position during inference, eliminating redundant `sincosf`/`powf` calls on every forward pass.

## Building

Prerequisites:
- CUDA toolkit (nvcc, cuBLAS, CUB)
- Qt5 (Widgets module)
- CMake 3.20+

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### Configuration

Edit `app_config.h` to set model path and default parameters:
```cpp
#define MODEL_DIR "path/to/Qwen3-0.6B"
```

Edit `config.h` to toggle INT8 quantization:
```cpp
constexpr bool USE_KV_CACHE_INT8 = true;  // or false for original BF16
```

## Project Structure

```
qwen600.qt/
├── CMakeLists.txt              Build configuration
├── config.h                    Model constants & INT8 toggle
├── app_config.h                Runtime defaults (model path, sampling params)
├── qwen_model.cuh              CUDA kernels (attention, quantization, RoPE, etc.)
├── static_loader.h             Safetensors weight loading via mmap
├── tokenizer.h                 BPE tokenizer
├── sampler.h                   Top-k / top-p / temperature sampling
├── inference_engine.h          C++ inference API (PIMPL)
├── inference_engine_impl.cu    CUDA inference implementation
├── mainwindow.h/cpp            Qt chat window
└── main.cpp                    Application entry point
```

## Based On

- Original engine: [qwen600.cu by yassa9](https://github.com/yassa9/qwen600)
- Model: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## License

MIT

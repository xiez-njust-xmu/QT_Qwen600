# qwen600.qt

基于 [qwen600.cu](https://github.com/yassa9/qwen600) 构建的桌面聊天应用，在原始 CUDA 推理引擎基础上增加了 Qt 图形界面、KV Cache INT8 量化以及内核级优化。

## 本项目的工作内容

基于原始 `qwen600.cu`（一个面向 Qwen3-0.6B 的极简 BF16 CUDA 推理引擎），本 fork 引入了以下改进：

| 特性 | 说明 |
|------|------|
| Qt 桌面图形界面 | 实时流式聊天界面，支持参数调节 |
| KV Cache INT8 量化 | 通过 per-head 对称量化，KV 缓存显存降低约 50% |
| Softmax Kernel 修复 | 修正 softmax 调度器，支持多种 block size 选择 |
| RoPE 预计算表 | 预计算 sin/cos 查找表，替代逐步在线计算 |

## Qt 聊天界面

使用原生 Qt 桌面窗口替代原始 CLI 交互，功能包括：

- Token 级实时流式显示
- 后台推理线程（UI 不阻塞）
- 可调生成参数：temperature、top-p、top-k
- 可配置 system prompt
- 停止生成 / 清除上下文按钮
- 深色主题（Catppuccin 风格配色）

**新增文件：** `mainwindow.h`、`mainwindow.cpp`、`main.cpp`、`app_config.h`、`inference_engine.h`、`inference_engine_impl.cu`

## KV Cache INT8 量化

### 问题

BF16 格式的 KV 缓存在 32K 上下文长度下占用约 3.5 GB 显存——在 8 GB 显卡上几乎占据一半预算。

### 解决方案

Per-token per-head 对称 INT8 量化，attention 计算时实时反量化。

- K/V 向量在完成 QK-Norm + RoPE（BF16 精度）后量化：`scale = max(|head|) / 127`
- Attention kernel 在 QK 点积和 V 加权求和时实时反量化 INT8 缓存
- 通过编译期开关控制：`config.h` 中 `constexpr bool USE_KV_CACHE_INT8`

### 显存对比

| 组件 | BF16（原始） | INT8（本工作） |
|------|-------------|---------------|
| K 缓存 | 1.75 GB | 875 MB |
| V 缓存 | 1.75 GB | 875 MB |
| Scale 数组 | - | 57.4 MB |
| **合计** | **3.5 GB** | **~1.8 GB** |

### 新增 CUDA Kernel

- `quantize_kv_head_kernel` — 基于 `cub::BlockReduce` 的 per-head absmax INT8 量化
- `attention_qk_kernel_int8` — QK 点积中实时反量化
- `attention_v_kernel_int8` — V 加权求和中实时反量化

## 内核优化

### Softmax Kernel 修复与调度

修正了 softmax 实现，采用多 block size 调度器根据序列长度选择最优线程数（64/256/512/1024），使用 `cub::BlockReduce` 实现数值稳定的并行 max-reduce 和 sum-reduce。

### RoPE 预计算表

用预计算的 cos/sin 查找表（`rope_precompute_init`）替代逐 token 在线 RoPE 计算。该表在模型初始化时一次性计算，推理时通过位置索引查表，消除了每次 forward pass 中的冗余 `sincosf`/`powf` 调用。

## 编译

前置条件：
- CUDA 工具链（nvcc、cuBLAS、CUB）
- Qt5（Widgets 模块）
- CMake 3.20+

```bash
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### 配置

编辑 `app_config.h` 设置模型路径和默认参数：
```cpp
#define MODEL_DIR "path/to/Qwen3-0.6B"
```

编辑 `config.h` 切换 INT8 量化：
```cpp
constexpr bool USE_KV_CACHE_INT8 = true;  // 或 false 恢复原始 BF16
```

## 项目结构

```
qwen600.qt/
├── CMakeLists.txt              构建配置
├── config.h                    模型常量 & INT8 开关
├── app_config.h                运行时默认值（模型路径、采样参数）
├── qwen_model.cuh              CUDA 内核（attention、量化、RoPE 等）
├── static_loader.h             基于 mmap 的 safetensors 权重加载
├── tokenizer.h                 BPE 分词器
├── sampler.h                   Top-k / top-p / temperature 采样
├── inference_engine.h          C++ 推理 API（PIMPL 模式）
├── inference_engine_impl.cu    CUDA 推理实现
├── mainwindow.h/cpp            Qt 聊天窗口
└── main.cpp                    应用入口
```

## 基于

- 原始引擎：[qwen600.cu by yassa9](https://github.com/yassa9/qwen600)
- 模型：[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)

## 许可证

MIT

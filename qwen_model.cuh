// qwen_model.cuh

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#include "config.h"
#include "static_loader.h"

// ================================================================
// globals
// ================================================================

#define EXIT_SUCCESS 0
constexpr int THREADS_PER_BLOCK = 256;

using bf16 = __nv_bfloat16;
using TransformerWeights = qwen_loader::QwenWeights;

// ================================================================
// transformer model
// ================================================================

typedef struct
{
    bf16 *x;       // activation at current time stamp (DIM,)
    bf16 *xb;      // buffer for residual branch (DIM,)
    bf16 *xb2;     // an additional buffer (DIM,)
    bf16 *hb;      // buffer for hidden dimension in ffn (HIDDEN_DIM,)
    bf16 *hb2;     // buffer for hidden dimension in ffn (HIDDEN_DIM,)
    bf16 *q;       // query buffer (Q_DIM,) - NOTE: This is larger than DIM now

    float *att;    // buffer for scores/attention values (N_HEADS, SEQ_LEN)
    bf16 *logits;  // output logits on the GPU (VOCAB_SIZE,)

    // kv cache (BF16, used when USE_KV_CACHE_INT8 == false)
    bf16* key_cache;   // (N_LAYERS, SEQ_LEN, KV_DIM)
    bf16* value_cache; // (N_LAYERS, SEQ_LEN, KV_DIM)

    // kv cache INT8 quantized (used when USE_KV_CACHE_INT8 == true)
    int8_t* key_cache_q;          // (N_LAYERS, SEQ_LEN, KV_DIM)
    int8_t* value_cache_q;        // (N_LAYERS, SEQ_LEN, KV_DIM)
    float*  key_cache_scale;      // (N_LAYERS, SEQ_LEN, N_KV_HEADS)
    float*  value_cache_scale;    // (N_LAYERS, SEQ_LEN, N_KV_HEADS)

    // temp BF16 buffers for current token K/V before quantization
    bf16* k_temp;  // (KV_DIM,)
    bf16* v_temp;  // (KV_DIM,)
    
    // buffer for final logits converted to fp32 on the GPU
    float* d_logits_fp32;

    // pre-computed RoPE cos/sin table (SEQ_LEN, HEAD_DIM/2)
    float* rope_cos;
    float* rope_sin;
} RunState;

typedef struct
{
    TransformerWeights weights;
    RunState state;

    cublasHandle_t cublas_handle;

    // host-side buffer to copy the final logits back for sampling
    float* h_logits;
} Transformer;


void
malloc_run_state(RunState* s)
{
    cudaMalloc(&s->x, DIM * sizeof(bf16));
    cudaMalloc(&s->xb, DIM * sizeof(bf16));
    cudaMalloc(&s->xb2, DIM * sizeof(bf16));
    cudaMalloc(&s->hb, HIDDEN_DIM * sizeof(bf16));
    cudaMalloc(&s->hb2, HIDDEN_DIM * sizeof(bf16));
    // query buffer must be Q_DIM, which is N_HEADS * HEAD_DIM = 2048 for this model.
    cudaMalloc(&s->q, Q_DIM * sizeof(bf16));
    
    cudaMalloc(&s->att, (size_t)N_HEADS * SEQ_LEN * sizeof(float));
    cudaMalloc(&s->logits, VOCAB_SIZE * sizeof(bf16));

    if constexpr (USE_KV_CACHE_INT8) {
        cudaMalloc(&s->key_cache_q,       (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(int8_t));
        cudaMalloc(&s->value_cache_q,     (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(int8_t));
        cudaMalloc(&s->key_cache_scale,   (size_t)N_LAYERS * SEQ_LEN * N_KV_HEADS * sizeof(float));
        cudaMalloc(&s->value_cache_scale, (size_t)N_LAYERS * SEQ_LEN * N_KV_HEADS * sizeof(float));
        cudaMalloc(&s->k_temp, KV_DIM * sizeof(bf16));
        cudaMalloc(&s->v_temp, KV_DIM * sizeof(bf16));
        s->key_cache = nullptr;
        s->value_cache = nullptr;
    } else {
        cudaMalloc(&s->key_cache,   (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));
        cudaMalloc(&s->value_cache, (size_t)N_LAYERS * SEQ_LEN * KV_DIM * sizeof(bf16));
        s->key_cache_q = nullptr;
        s->value_cache_q = nullptr;
        s->key_cache_scale = nullptr;
        s->value_cache_scale = nullptr;
        s->k_temp = nullptr;
        s->v_temp = nullptr;
    }

    cudaMalloc(&s->d_logits_fp32, VOCAB_SIZE * sizeof(float));

    constexpr size_t rope_table_size = (size_t)SEQ_LEN * (HEAD_DIM / 2);
    cudaMalloc(&s->rope_cos, rope_table_size * sizeof(float));
    cudaMalloc(&s->rope_sin, rope_table_size * sizeof(float));
}

void rope_precompute_init(float* rope_cos, float* rope_sin);

void
build_transformer(
    Transformer* t,
    const char* checkpoint_path)
{
    qwen_loader::load_qwen_weights(checkpoint_path, t->weights);
    malloc_run_state(&t->state);

    rope_precompute_init(t->state.rope_cos, t->state.rope_sin);

    cudaMallocHost((void**)&t->h_logits, VOCAB_SIZE * sizeof(float));

    cublasCreate(&t->cublas_handle);
}

void
free_transformer(Transformer* t)
{
    cudaFree(t->state.x);
    cudaFree(t->state.xb);
    cudaFree(t->state.xb2);
    cudaFree(t->state.hb);
    cudaFree(t->state.hb2);
    cudaFree(t->state.q);
    cudaFree(t->state.att);
    cudaFree(t->state.logits);

    if constexpr (USE_KV_CACHE_INT8) {
        cudaFree(t->state.key_cache_q);
        cudaFree(t->state.value_cache_q);
        cudaFree(t->state.key_cache_scale);
        cudaFree(t->state.value_cache_scale);
        cudaFree(t->state.k_temp);
        cudaFree(t->state.v_temp);
    } else {
        cudaFree(t->state.key_cache);
        cudaFree(t->state.value_cache);
    }

    cudaFree(t->state.d_logits_fp32);
    cudaFree(t->state.rope_cos);
    cudaFree(t->state.rope_sin);

    cudaFreeHost(t->h_logits);
    cublasDestroy(t->cublas_handle);
}

// ================================================================
// CUDA OPTIMIZED KERNELS 
// ================================================================
// RMS Norm
// ================================================================
template <int THREADS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
rms_norm_kernel(
    __nv_bfloat16* __restrict__ Y,
    const __nv_bfloat16* __restrict__ X,
    const __nv_bfloat16* __restrict__ weight,
    size_t D)
{
    const int t_idx = threadIdx.x;
    const int vec_iters = D / 2;

    const __nv_bfloat162* row_in = reinterpret_cast<const __nv_bfloat162*>(X);
    const __nv_bfloat162* weight_in = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* row_out = reinterpret_cast<__nv_bfloat162*>(Y);

    float lsum = 0.0f;

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK)
    {
        __nv_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        // convert to fp32 for math
        float2 v_fp32 = __bfloat1622float2(v_bf16);

        // lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
        lsum = __fmaf_rn(v_fp32.x, v_fp32.x, lsum);
        lsum = __fmaf_rn(v_fp32.y, v_fp32.y, lsum);
    }

    using BlockReduce = cub::BlockReduce<float, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_sum = BlockReduce(tmp).Sum(lsum);

    __shared__ float mul_val;
    if (t_idx == 0)
    {
        float val = __fmaf_rn(block_sum, INV_DIM, EPS);
        float approx = __frsqrt_rn(val);
        // mul_val = approx * (1.5f - 0.5f * val * approx * approx);
        mul_val = rsqrtf(val);
    }
    __syncthreads();

    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK)
    {
        __nv_bfloat162 v_in_bf16 = __ldg(&row_in[idx]);
        __nv_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32 = __bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32 = __bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        // convert back to BF16 for storing
        row_out[idx] = __float22bfloat162_rn(v_in_fp32);
    }
}

void
rmsnorm_gpu(
    __nv_bfloat16* o,
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    int dim)
{
    if (dim % 2 != 0)
    {
        fprintf(stderr, "FATAL: rmsnorm dim %d is not divisible by 2. Vectorized kernel cannot run.\n", dim);
        exit(EXIT_FAILURE);
    }
    // if dim > (THREADS_PER_BLOCK * some_threshold), a multi-block reduction might be needed,
    // but for typical dimensions up to 8192, a single block is sufficient and simpler.
    const int num_blocks = 1;

    rms_norm_kernel<THREADS_PER_BLOCK><<<num_blocks, THREADS_PER_BLOCK>>>
        (o, x, weight, dim);
}

template <int THREADS_PER_BLOCK, int HEAD_DIM>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_multi_rmsnorm_kernel(
    bf16* __restrict__ vecs,
    const bf16* __restrict__ weight,
    int num_vecs)
{
    // each block processes one vector/head
    const int vec_idx = blockIdx.x;
    if (vec_idx >= num_vecs) return;

    const int t_idx = threadIdx.x;
    const int vec_iters = HEAD_DIM / 2;

    bf16* vec_start = vecs + vec_idx * HEAD_DIM;

    const __nv_bfloat162* row_in = reinterpret_cast<const __nv_bfloat162*>(vec_start);
    const __nv_bfloat162* weight_in = reinterpret_cast<const __nv_bfloat162*>(weight);
    __nv_bfloat162* row_out = reinterpret_cast<__nv_bfloat162*>(vec_start);

    // 1. calculate sum of squares
    float lsum = 0.0f;
    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK)
    {
        __nv_bfloat162 v_bf16 = __ldg(&row_in[idx]);
        float2 v_fp32 = __bfloat1622float2(v_bf16);
        lsum += v_fp32.x * v_fp32.x + v_fp32.y * v_fp32.y;
    }

    // 2. reduce sum within the block
    using BlockReduce = cub::BlockReduce<float, THREADS_PER_BLOCK>;
    __shared__ typename BlockReduce::TempStorage tmp;
    float block_sum = BlockReduce(tmp).Sum(lsum);

    // 3. calculate the normalization factor
    __shared__ float mul_val;
    if (t_idx == 0) { mul_val = rsqrtf(block_sum * INV_HEAD_DIM + EPS); }
    __syncthreads();

    // 4. applying the normalization
    for (int idx = t_idx; idx < vec_iters; idx += THREADS_PER_BLOCK)
    {
        __nv_bfloat162 v_in_bf16 = __ldg(&row_in[idx]);
        __nv_bfloat162 v_weight_bf16 = __ldg(&weight_in[idx]);
        float2 v_in_fp32 = __bfloat1622float2(v_in_bf16);
        float2 v_weight_fp32 = __bfloat1622float2(v_weight_bf16);

        v_in_fp32.x = (v_in_fp32.x * mul_val) * v_weight_fp32.x;
        v_in_fp32.y = (v_in_fp32.y * mul_val) * v_weight_fp32.y;

        row_out[idx] = __float22bfloat162_rn(v_in_fp32);
    }
}

void
qk_norm_fused_gpu(
    bf16* q,
    bf16* k,
    const bf16* q_norm_weight,
    const bf16* k_norm_weight)
{
    constexpr int QK_NORM_THREADS_PER_BLOCK = 64;

    // launching ONE kernel for all query heads
    fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM><<<N_HEADS, QK_NORM_THREADS_PER_BLOCK>>>
    (q, q_norm_weight, N_HEADS);

    // launching ONE kernel for all key heads
    fused_multi_rmsnorm_kernel<QK_NORM_THREADS_PER_BLOCK, HEAD_DIM><<<N_KV_HEADS, QK_NORM_THREADS_PER_BLOCK>>>
    (k, k_norm_weight, N_KV_HEADS);
}

// ================================================================
// RoPE
// ================================================================
__global__ void
rope_kernel(
    __nv_bfloat16* __restrict__ q, 
    __nv_bfloat16* __restrict__ k, 
    int pos) 
{
    // grid: Q_DIM / 2, block: THREADS_PER_BLOCK
    // each thread handles one pair of dimensions (i, i+1)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Q_DIM / 2) { return; }

    int head_dim_idx = (i * 2) % HEAD_DIM;
    float freq = 1.0f / powf(ROPE_THETA, (float)head_dim_idx / (float)HEAD_DIM);
    float val = (float)pos * freq;
    float fcr, fci;
    sincosf(val, &fci, &fcr);

    // rotate Q
    __nv_bfloat162 q_val_bf16 = reinterpret_cast<__nv_bfloat162*>(q)[i];
    float2 q_val_fp32 = __bfloat1622float2(q_val_bf16);
    float q0 = q_val_fp32.x * fcr - q_val_fp32.y * fci;
    float q1 = q_val_fp32.x * fci + q_val_fp32.y * fcr;
    reinterpret_cast<__nv_bfloat162*>(q)[i] = __float22bfloat162_rn(make_float2(q0, q1));

    if (i < KV_DIM / 2)
    {
        // rotate K
        __nv_bfloat162 k_val_bf16 = reinterpret_cast<__nv_bfloat162*>(k)[i];
        float2 k_val_fp32 = __bfloat1622float2(k_val_bf16);
        float k0 = k_val_fp32.x * fcr - k_val_fp32.y * fci;
        float k1 = k_val_fp32.x * fci + k_val_fp32.y * fcr;
        reinterpret_cast<__nv_bfloat162*>(k)[i] = __float22bfloat162_rn(make_float2(k0, k1));
    }
}
    
void 
rope_gpu(
    __nv_bfloat16* q, 
    __nv_bfloat16* k, 
    int pos)
{
    int num_pairs = Q_DIM / 2;
    int grid_size = (num_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rope_kernel<<<grid_size, THREADS_PER_BLOCK>>>(q, k, pos);
}

// ================================================================
// RoPE Pre-computed Table
// ================================================================
__global__ void
rope_precompute_kernel(
    float* __restrict__ rope_cos,
    float* __restrict__ rope_sin)
{
    int pos = blockIdx.x;
    int j = threadIdx.x;

    if (j < HEAD_DIM / 2)
    {
        float freq = 1.0f / powf(ROPE_THETA, (float)(j * 2) / (float)HEAD_DIM);
        float val = (float)pos * freq;
        float sin_val, cos_val;
        sincosf(val, &sin_val, &cos_val);

        int idx = pos * (HEAD_DIM / 2) + j;
        rope_cos[idx] = cos_val;
        rope_sin[idx] = sin_val;
    }
}

void
rope_precompute_init(float* rope_cos, float* rope_sin)
{
    dim3 grid(SEQ_LEN);
    dim3 block(HEAD_DIM / 2);
    rope_precompute_kernel<<<grid, block>>>(rope_cos, rope_sin);
    cudaDeviceSynchronize();
}

__global__ void
rope_table_kernel(
    bf16* __restrict__ q,
    bf16* __restrict__ k_cache_pos,
    const float* __restrict__ rope_cos,
    const float* __restrict__ rope_sin,
    int pos)
{
    int h = blockIdx.x;
    int j = threadIdx.x;

    if (j >= HEAD_DIM / 2) return;

    int table_idx = pos * (HEAD_DIM / 2) + j;
    float fcr = rope_cos[table_idx];
    float fci = rope_sin[table_idx];

    if (h < N_HEADS)
    {
        bf16* q_head = q + h * HEAD_DIM;

        float q_real = __bfloat162float(q_head[j]);
        float q_imag = __bfloat162float(q_head[j + HEAD_DIM / 2]);

        float q_rotated_real = q_real * fcr - q_imag * fci;
        float q_rotated_imag = q_real * fci + q_imag * fcr;

        q_head[j]              = __float2bfloat16_rn(q_rotated_real);
        q_head[j + HEAD_DIM/2] = __float2bfloat16_rn(q_rotated_imag);
    }

    if (h < N_KV_HEADS)
    {
        bf16* k_head = k_cache_pos + h * HEAD_DIM;

        float k_real = __bfloat162float(k_head[j]);
        float k_imag = __bfloat162float(k_head[j + HEAD_DIM / 2]);

        float k_rotated_real = k_real * fcr - k_imag * fci;
        float k_rotated_imag = k_real * fci + k_imag * fcr;

        k_head[j]              = __float2bfloat16_rn(k_rotated_real);
        k_head[j + HEAD_DIM/2] = __float2bfloat16_rn(k_rotated_imag);
    }
}

void
rope_gpu_table(
    bf16* q,
    bf16* k_cache_pos,
    const float* rope_cos,
    const float* rope_sin,
    int pos)
{
    dim3 grid(N_HEADS);
    dim3 block(HEAD_DIM / 2);
    rope_table_kernel<<<grid, block>>>(q, k_cache_pos, rope_cos, rope_sin, pos);
}

// ================================================================
// softmax
// ================================================================
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE)
softmax_kernel_parallel(
    float* att,
    int pos)
{
    int h = blockIdx.x;
    int tid = threadIdx.x;

    float* scores = att + (size_t)h * SEQ_LEN;
    int len = pos + 1;

    // 1. parallel max reduction
    float local_max = -1e9f;
    for (int i = tid; i < len; i += BLOCK_SIZE)
    {
        float v = scores[i];
        if (v > local_max) { local_max = v; }
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_max;
    __shared__ float shared_sum;

    float block_max = BlockReduce(temp_storage).Reduce(local_max, cub::Max());
    if (tid == 0) { shared_max = block_max; }
    __syncthreads();

    float max_val = shared_max;

    // 2. parallel exp and sum
    float local_sum = 0.0f;
    for (int i = tid; i < len; i += BLOCK_SIZE)
    {
        float e = expf(scores[i] - max_val);
        scores[i] = e;
        local_sum += e;
    }
    __syncthreads();

    float block_sum = BlockReduce(temp_storage).Sum(local_sum);
    if (tid == 0) { shared_sum = block_sum; }
    __syncthreads();

    // 3. parallel normalize
    float inv_sum = 1.0f / shared_sum;
    for (int i = tid; i < len; i += BLOCK_SIZE)
    {
        scores[i] *= inv_sum;
    }
}

void
softmax_gpu(float* att, int pos)
{
    int len = pos + 1;
    if (len <= 64)
    {
        softmax_kernel_parallel<64><<<N_HEADS, 64>>>(att, pos);
    }
    else if (len <= 256)
    {
        softmax_kernel_parallel<256><<<N_HEADS, 256>>>(att, pos);
    }
    else if (len <= 512)
    {
        softmax_kernel_parallel<512><<<N_HEADS, 512>>>(att, pos);
    }
    else
    {
        softmax_kernel_parallel<1024><<<N_HEADS, 1024>>>(att, pos);
    }
}

// ================================================================
// Attention
// ================================================================
__global__ void
attention_qk_kernel(
    float* att,
    const bf16* q,
    const bf16* k_cache,
    int pos)
{
    // grid: N_HEADS, block: pos + 1 (up to 1024)
    int h = blockIdx.x; 
    int t = threadIdx.x;

    constexpr int kv_mul = N_HEADS / N_KV_HEADS;

    if (t <= pos)
    {
        const bf16* q_head = q + h * HEAD_DIM;
        int kv_head_idx = h / kv_mul;
        const bf16* k_vec = k_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM / 2; i++)
        {
            __nv_bfloat162 q_pair = reinterpret_cast<const __nv_bfloat162*>(q_head)[i];
            __nv_bfloat162 k_pair = reinterpret_cast<const __nv_bfloat162*>(k_vec)[i];

            float2 q_vals = __bfloat1622float2(q_pair);
            float2 k_vals = __bfloat1622float2(k_pair);

            // score += q_vals.x * k_vals.x + q_vals.y * k_vals.y;
            score = __fmaf_rn(q_vals.x, k_vals.x, score);
            score = __fmaf_rn(q_vals.y, k_vals.y, score);
        }

        score /= sqrtf((float)HEAD_DIM);
        att[(size_t)h * SEQ_LEN + t] = score;
}

}
__global__ void
attention_v_kernel(
    bf16* out,
    const float* att,
    const bf16* v_cache,
    int pos)
{
    // grid: N_HEADS, block: HEAD_DIM
    int h = blockIdx.x;
    int i = threadIdx.x; // idx within the head dimension
    constexpr int kv_mul = N_HEADS / N_KV_HEADS;

    bf16* out_head = out + (size_t)h * HEAD_DIM;
    const float* att_head = att + (size_t)h * SEQ_LEN;
    int kv_head_idx = h / kv_mul;

    float weighted_sum = 0.0f;
    for (int t = 0; t <= pos; t++)
    {
        const bf16* v_vec = v_cache + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        // weighted_sum += att_head[t] * __bfloat162float(v_vec[i]);   
        weighted_sum = __fmaf_rn(att_head[t], __bfloat162float(v_vec[i]), weighted_sum);
    }
    out_head[i] = __float2bfloat16_rn(weighted_sum);
}

// ================================================================
// KV Cache INT8 Quantization
// ================================================================
__global__ void
quantize_kv_head_kernel(
    int8_t* __restrict__ out_q,
    float*  __restrict__ out_scale,
    const bf16* __restrict__ in_bf16)
{
    int head = blockIdx.x;
    int elem = threadIdx.x;
    int global_idx = head * HEAD_DIM + elem;

    float val = __bfloat162float(in_bf16[global_idx]);
    float abs_val = fabsf(val);

    using BlockReduce = cub::BlockReduce<float, HEAD_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    float head_max = BlockReduce(temp).Reduce(abs_val, cub::Max());

    __shared__ float shared_scale;
    if (elem == 0) {
        float scale = (head_max > 0.0f) ? (head_max / 127.0f) : 1.0f;
        shared_scale = scale;
        out_scale[head] = scale;
    }
    __syncthreads();

    float scale = shared_scale;
    int quantized = __float2int_rn(val / scale);
    quantized = max(-127, min(127, quantized));
    out_q[global_idx] = (int8_t)quantized;
}

void
quantize_kv_to_int8(
    int8_t* out_q,
    float* out_scale,
    const bf16* in_bf16)
{
    quantize_kv_head_kernel<<<N_KV_HEADS, HEAD_DIM>>>(out_q, out_scale, in_bf16);
}

// ================================================================
// Attention INT8 Kernels
// ================================================================
__global__ void
attention_qk_kernel_int8(
    float* att,
    const bf16* q,
    const int8_t* k_cache_q,
    const float* k_cache_scale,
    int pos)
{
    int h = blockIdx.x;
    int t = threadIdx.x;
    constexpr int kv_mul = N_HEADS / N_KV_HEADS;

    if (t <= pos)
    {
        const bf16* q_head = q + h * HEAD_DIM;
        int kv_head_idx = h / kv_mul;
        float scale = k_cache_scale[(size_t)t * N_KV_HEADS + kv_head_idx];
        const int8_t* k_vec = k_cache_q + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;

        float score = 0.0f;
        for (int i = 0; i < HEAD_DIM; i += 4)
        {
            float q0 = __bfloat162float(q_head[i]);
            float q1 = __bfloat162float(q_head[i + 1]);
            float q2 = __bfloat162float(q_head[i + 2]);
            float q3 = __bfloat162float(q_head[i + 3]);

            float k0 = (float)k_vec[i]     * scale;
            float k1 = (float)k_vec[i + 1] * scale;
            float k2 = (float)k_vec[i + 2] * scale;
            float k3 = (float)k_vec[i + 3] * scale;

            score = __fmaf_rn(q0, k0, score);
            score = __fmaf_rn(q1, k1, score);
            score = __fmaf_rn(q2, k2, score);
            score = __fmaf_rn(q3, k3, score);
        }

        score /= sqrtf((float)HEAD_DIM);
        att[(size_t)h * SEQ_LEN + t] = score;
    }
}

__global__ void
attention_v_kernel_int8(
    bf16* out,
    const float* att,
    const int8_t* v_cache_q,
    const float* v_cache_scale,
    int pos)
{
    int h = blockIdx.x;
    int i = threadIdx.x;
    constexpr int kv_mul = N_HEADS / N_KV_HEADS;

    bf16* out_head = out + (size_t)h * HEAD_DIM;
    const float* att_head = att + (size_t)h * SEQ_LEN;
    int kv_head_idx = h / kv_mul;

    float weighted_sum = 0.0f;
    for (int t = 0; t <= pos; t++)
    {
        float scale = v_cache_scale[(size_t)t * N_KV_HEADS + kv_head_idx];
        const int8_t* v_vec = v_cache_q + (size_t)t * KV_DIM + (size_t)kv_head_idx * HEAD_DIM;
        float v_val = (float)v_vec[i] * scale;
        weighted_sum = __fmaf_rn(att_head[t], v_val, weighted_sum);
    }
    out_head[i] = __float2bfloat16_rn(weighted_sum);
}

void
attention_gpu(
    RunState* s,
    int l, 
    int pos)
{
    int qk_threads_per_block = std::min(1024, pos + 1);

    if constexpr (USE_KV_CACHE_INT8) {
        int8_t* layer_key_cache_q   = s->key_cache_q      + (size_t)l * SEQ_LEN * KV_DIM;
        int8_t* layer_value_cache_q = s->value_cache_q    + (size_t)l * SEQ_LEN * KV_DIM;
        float*  layer_key_scale     = s->key_cache_scale  + (size_t)l * SEQ_LEN * N_KV_HEADS;
        float*  layer_value_scale   = s->value_cache_scale + (size_t)l * SEQ_LEN * N_KV_HEADS;

        attention_qk_kernel_int8<<<N_HEADS, qk_threads_per_block>>>(
            s->att, s->q, layer_key_cache_q, layer_key_scale, pos
        );

        softmax_gpu(s->att, pos);

        attention_v_kernel_int8<<<N_HEADS, HEAD_DIM>>>(
            s->q, s->att, layer_value_cache_q, layer_value_scale, pos
        );
    } else {
        bf16* layer_key_cache   = s->key_cache   + (size_t)l * SEQ_LEN * KV_DIM;
        bf16* layer_value_cache = s->value_cache + (size_t)l * SEQ_LEN * KV_DIM;

        attention_qk_kernel<<<N_HEADS, qk_threads_per_block>>>(
            s->att, s->q, layer_key_cache, pos
        );

        softmax_gpu(s->att, pos);

        attention_v_kernel<<<N_HEADS, HEAD_DIM>>>(
            s->q, s->att, layer_value_cache, pos
        );
    }
}

// ================================================================
// swiGlu
// ================================================================
__global__ void
swiglu_kernel(
    __nv_bfloat16* hb, 
    const __nv_bfloat16* hb2, 
    int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        float val_fp32 = __bfloat162float(hb[i]);
        float hb2_fp32 = __bfloat162float(hb2[i]);
        
        float silu_val = val_fp32 * (1.0f / (1.0f + expf(-val_fp32)));
        float result_fp32 = silu_val * hb2_fp32;
        hb[i] = __float2bfloat16_rn(result_fp32);
    }
}

void
swiglu_gpu(
    __nv_bfloat16* hb, 
    const __nv_bfloat16* hb2, 
    int size)
{
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    swiglu_kernel<<<grid_size, THREADS_PER_BLOCK>>>(hb, hb2, size);
}

// ================================================================
// cuBLAS matmul
// ================================================================
// performs a matrix-vector multiplication y = Wx using cuBLAS.
// W is a matrix (m rows, n cols), x is a vector (len n), y is a vector (len m).
void
matmul_cublas(
    cublasHandle_t handle, 
    __nv_bfloat16* y, 
    const __nv_bfloat16* W, 
    const __nv_bfloat16* x, 
    int m, 
    int n,
    float alpha = 1.0f,
    float beta = 0.0f)
{
    // in cuBLAS, matrices are column-major by default. 
    // weights are row-major.
    // W matrix is (m, n) in row-major layout, which is (n, m) in column-major.
    // we want to compute y = Wx.
    // by telling cublasSgemv to use the transpose of W (CUBLAS_OP_T),
    // it correctly treats our row-major matrix as a row-major matrix.

    // C = alpha * (A @ B) + beta * C

    // cublasSgemv: y = alpha * op(A) * x + beta * y
    // op(A) is our W matrix. handle is the cuBLAS context.
    // CUBLAS_OP_T means "use the transpose of A".
    // n, m are the dimensions of the matrix as seen by cuBLAS (column-major).
    // So for our (m, n) row-major matrix, it's seen as (n, m) column-major.
    // W is the pointer to the matrix. n is the leading dimension (width of the row-major matrix).
    // x is the input vector.  1 is its stride.
    // y is the output vector. 1 is its stride.
    cublasGemmEx(handle,
                 CUBLAS_OP_T,        // Transpose W (since it's row-major)
                 CUBLAS_OP_N,        // Don't transpose x
                 m,                  // rows of C (output y)
                 1,                  // columns of C (output y is a vector)
                 n,                  // common dimension (k)
                 
                 &alpha,             // host pointer
                 W,                  // A matrix (W)
                 CUDA_R_16BF,        // A datatype
                 n,                  // leading dimension of A

                 x,                  // B matrix (x)
                 CUDA_R_16BF,        // B datatype
                 n,                  // leading dimension of B
                 
                 &beta,              // host pointer
                 y,                  // C matrix (y)
                 CUDA_R_16BF,        // C datatype
                 m,                  // leading dimension of C
                 
                 CUDA_R_32F,         // compute type: use fp32 for precision
                 CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// ================================================================
// forward 
// ================================================================
__global__ void
convert_bf16_to_fp32_kernel(
    __nv_bfloat16* bf16_in, 
    float* fp32_out, 
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n){ fp32_out[i] = __bfloat162float(bf16_in[i]); }
}

float* forward(
    Transformer* transformer, 
    int token, 
    int pos)
{
    RunState* s = &transformer->state;
    TransformerWeights* w = &transformer->weights;
    cublasHandle_t handle = transformer->cublas_handle;

    // 1. token embedding lookup
    // copy the token embedding into the main activation buffer s->x
    bf16* token_embedding_ptr = w->token_embedding_table + (size_t)token * DIM;
    cudaMemcpy(s->x, token_embedding_ptr, (size_t)DIM * sizeof(bf16), cudaMemcpyDeviceToDevice);

    for (int l = 0; l < N_LAYERS; l++)
    {
        const qwen_loader::TransformerBlockWeights& layer = w->layers[l];

        // === ATTENTION BLOCK ===

        // 2. input RMSNorm (pre-attention norm)
        rmsnorm_gpu(s->xb, s->x, layer.input_layernorm_weight, DIM);

        // 3. QKV matrix multiplications
        if constexpr (USE_KV_CACHE_INT8) {
            matmul_cublas(handle, s->q,      layer.attention.q_proj_weight, s->xb,  Q_DIM, DIM);
            matmul_cublas(handle, s->k_temp, layer.attention.k_proj_weight, s->xb, KV_DIM, DIM);
            matmul_cublas(handle, s->v_temp, layer.attention.v_proj_weight, s->xb, KV_DIM, DIM);

            // 4. QK-Norm (in BF16 on temp buffer)
            qk_norm_fused_gpu(s->q, s->k_temp, layer.attention.q_norm_weight, layer.attention.k_norm_weight);

            // 5. RoPE (in BF16 on temp buffer)
            rope_gpu_table(s->q, s->k_temp, s->rope_cos, s->rope_sin, pos);

            // 6. Quantize K and V into INT8 cache
            int8_t* k_cache_pos_q = s->key_cache_q   + (size_t)l * SEQ_LEN * KV_DIM + (size_t)pos * KV_DIM;
            float*  k_scale_pos   = s->key_cache_scale + (size_t)l * SEQ_LEN * N_KV_HEADS + (size_t)pos * N_KV_HEADS;
            quantize_kv_to_int8(k_cache_pos_q, k_scale_pos, s->k_temp);

            int8_t* v_cache_pos_q = s->value_cache_q   + (size_t)l * SEQ_LEN * KV_DIM + (size_t)pos * KV_DIM;
            float*  v_scale_pos   = s->value_cache_scale + (size_t)l * SEQ_LEN * N_KV_HEADS + (size_t)pos * N_KV_HEADS;
            quantize_kv_to_int8(v_cache_pos_q, v_scale_pos, s->v_temp);
        } else {
            bf16* k_cache_pos = s->key_cache   + (size_t)l * SEQ_LEN * KV_DIM + (size_t)pos * KV_DIM;
            bf16* v_cache_pos = s->value_cache + (size_t)l * SEQ_LEN * KV_DIM + (size_t)pos * KV_DIM;
            matmul_cublas(handle, s->q,        layer.attention.q_proj_weight, s->xb,  Q_DIM, DIM);
            matmul_cublas(handle, k_cache_pos, layer.attention.k_proj_weight, s->xb, KV_DIM, DIM);
            matmul_cublas(handle, v_cache_pos, layer.attention.v_proj_weight, s->xb, KV_DIM, DIM);

            // 4. QK-Norm
            qk_norm_fused_gpu(s->q, k_cache_pos, layer.attention.q_norm_weight, layer.attention.k_norm_weight);

            // 5. RoPE (table-based, pre-computed sin/cos)
            rope_gpu_table(s->q, k_cache_pos, s->rope_cos, s->rope_sin, pos);
        }

        // 6. MHA (QK^T V)
        attention_gpu(s, l, pos);

        // 7. final attention output projection and residual connection
        matmul_cublas(handle, s->x, layer.attention.o_proj_weight, s->q, DIM, Q_DIM, 1.0f, 1.0f);

        // === FFN BLOCK ===

        // 8. post-attention RMSNorm
        rmsnorm_gpu(s->xb, s->x, layer.post_attention_layernorm_weight, DIM);

        // 9. FFN projections (Gate and Up)
        // output of w1 matmul is s->hb. output of w3 matmul is s->hb2.
        matmul_cublas(handle, s->hb,  layer.ffn.gate_proj_weight, s->xb, HIDDEN_DIM, DIM);
        matmul_cublas(handle, s->hb2, layer.ffn.up_proj_weight,   s->xb, HIDDEN_DIM, DIM);

        // 9. SwiGLU
        // in-place operation on s->hb, using s->hb2 as the gate.
        swiglu_gpu(s->hb, s->hb2, HIDDEN_DIM);

        // 10. final FFN Down Projection matmul and residual connection
        matmul_cublas(handle, s->x, layer.ffn.down_proj_weight, s->hb, DIM, HIDDEN_DIM, 1.0f, 1.0f);
    }

    // === FINAL CLASSIFIER ===

    // 11. final RMSNorm
    // in-place operation on s->x
    rmsnorm_gpu(s->x, s->x, w->final_norm_weight, DIM);

    // 12. classifier Matmul
    matmul_cublas(handle, s->logits, w->output_head_weight, s->x, VOCAB_SIZE, DIM);

    int grid_size = (VOCAB_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    convert_bf16_to_fp32_kernel<<<grid_size, THREADS_PER_BLOCK>>>(s->logits, s->d_logits_fp32, VOCAB_SIZE);

    // 13. copy the fp32 logits from GPU device to pinned host memory for the CPU to access
    cudaMemcpy(transformer->h_logits, s->d_logits_fp32, (size_t)VOCAB_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    return transformer->h_logits;
}

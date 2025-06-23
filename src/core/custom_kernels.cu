/**
 * High-Performance Custom CUDA Kernels
 * Optimized implementations for LLM inference
 */
#include "custom_kernels.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cutlass/cutlass.h>
#include <cooperative_groups.h>
namespace openinferencev2 {
// FlashAttention implementation
__global__ void flash_attention_kernel(
    const float* Q,  // [batch_size, num_heads, seq_len, head_dim]
    const float* K,  // [batch_size, num_heads, seq_len, head_dim] 
    const float* V,  // [batch_size, num_heads, seq_len, head_dim]
    float* O,        // [batch_size, num_heads, seq_len, head_dim]
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    // Shared memory for tiles
    extern __shared__ float shmem[];
    float* Q_tile = shmem;
    float* K_tile = Q_tile + blockDim.x * head_dim;
    float* V_tile = K_tile + blockDim.y * head_dim;
    float* S_tile = V_tile + blockDim.y * head_dim;
    
    const int batch_idx = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (seq_idx >= seq_len) return;
    
    // Load Q tile into shared memory
    for (int d = threadIdx.y; d < head_dim; d += blockDim.y) {
        int q_idx = batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim +
                   seq_idx * head_dim + d;
        Q_tile[threadIdx.x * head_dim + d] = Q[q_idx];
    }
    
    __syncthreads();
    
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float acc[head_dim] = {0.0f};
    
    // Process K,V tiles
    for (int k_start = 0; k_start < seq_len; k_start += blockDim.y) {
        // Load K,V tiles
        for (int k_idx = threadIdx.y; k_idx < min(blockDim.y, seq_len - k_start); k_idx += blockDim.y) {
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                int global_k_idx = k_start + k_idx;
                int k_mem_idx = batch_idx * num_heads * seq_len * head_dim +
                               head_idx * seq_len * head_dim +
                               global_k_idx * head_dim + d;
                               
                K_tile[k_idx * head_dim + d] = K[k_mem_idx];
                V_tile[k_idx * head_dim + d] = V[k_mem_idx];
            }
        }
        
        __syncthreads();
        
        // Compute attention scores
        for (int k_idx = 0; k_idx < min(blockDim.y, seq_len - k_start); ++k_idx) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += Q_tile[threadIdx.x * head_dim + d] * K_tile[k_idx * head_dim + d];
            }
            score *= scale;
            
            // Online softmax computation
            float new_max = fmaxf(row_max, score);
            float exp_score = expf(score - new_max);
            float exp_old_max = expf(row_max - new_max);
            
            // Update running sum
            row_sum = row_sum * exp_old_max + exp_score;
            row_max = new_max;
            
            // Update accumulated output
            float weight = exp_score;
            for (int d = 0; d < head_dim; ++d) {
                acc[d] = acc[d] * exp_old_max + weight * V_tile[k_idx * head_dim + d];
            }
        }
        
        __syncthreads();
    }
    
    // Normalize and write output
    for (int d = 0; d < head_dim; ++d) {
        int o_idx = batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim +
                   seq_idx * head_dim + d;
        O[o_idx] = acc[d] / row_sum;
    }
}
void flash_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream
) {
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    dim3 block(32, 32);  // 32x32 threads per block
    dim3 grid((seq_len + block.x - 1) / block.x, num_heads, batch_size);
    
    size_t shmem_size = (2 * block.x + 2 * block.y) * head_dim * sizeof(float);
    
    flash_attention_kernel<<<grid, block, shmem_size, stream>>>(
        Q, K, V, O, batch_size, num_heads, seq_len, head_dim, scale
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("FlashAttention kernel launch failed: " + 
                           std::string(cudaGetErrorString(error)));
    }
}
// Fused Feed-Forward Network kernel
__global__ void fused_ffn_kernel(
    const float* input,      // [batch_size, seq_len, hidden_size]
    const float* gate_weight, // [hidden_size, intermediate_size]
    const float* up_weight,   // [hidden_size, intermediate_size]  
    const float* down_weight, // [intermediate_size, hidden_size]
    float* output,           // [batch_size, seq_len, hidden_size]
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const int intermediate_size
) {
    const int batch_idx = blockIdx.z;
    const int seq_idx = blockIdx.y;
    const int hidden_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (hidden_idx >= hidden_size) return;
    
    extern __shared__ float shmem[];
    float* input_cache = shmem;
    float* gate_cache = input_cache + hidden_size;
    float* up_cache = gate_cache + intermediate_size;
    
    // Load input vector into shared memory
    if (threadIdx.x < hidden_size) {
        int input_idx = batch_idx * seq_len * hidden_size + 
                       seq_idx * hidden_size + threadIdx.x;
        input_cache[threadIdx.x] = input[input_idx];
    }
    
    __syncthreads();
    
    // Compute gate and up projections
    for (int inter_idx = threadIdx.x; inter_idx < intermediate_size; inter_idx += blockDim.x) {
        float gate_sum = 0.0f;
        float up_sum = 0.0f;
        
        for (int h = 0; h < hidden_size; ++h) {
            float input_val = input_cache[h];
            gate_sum += input_val * gate_weight[h * intermediate_size + inter_idx];
            up_sum += input_val * up_weight[h * intermediate_size + inter_idx];
        }
        
        // Apply SiLU activation to gate
        float gate_activated = gate_sum / (1.0f + expf(-gate_sum));
        
        // Element-wise multiplication
        gate_cache[inter_idx] = gate_activated * up_sum;
    }
    
    __syncthreads();
    
    // Down projection
    if (hidden_idx < hidden_size) {
        float output_sum = 0.0f;
        
        for (int inter_idx = 0; inter_idx < intermediate_size; ++inter_idx) {
            output_sum += gate_cache[inter_idx] * down_weight[inter_idx * hidden_size + hidden_idx];
        }
        
        int output_idx = batch_idx * seq_len * hidden_size + 
                        seq_idx * hidden_size + hidden_idx;
        output[output_idx] = output_sum;
    }
}
void fused_ffn_forward(
    const float* input, const float* gate_weight, const float* up_weight,
    const float* down_weight, float* output,
    int batch_size, int seq_len, int hidden_size, int intermediate_size,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid((hidden_size + block.x - 1) / block.x, seq_len, batch_size);
    
    size_t shmem_size = (hidden_size + 2 * intermediate_size) * sizeof(float);
    
    fused_ffn_kernel<<<grid, block, shmem_size, stream>>>(
        input, gate_weight, up_weight, down_weight, output,
        batch_size, seq_len, hidden_size, intermediate_size
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("Fused FFN kernel launch failed: " + 
                           std::string(cudaGetErrorString(error)));
    }
}
// Optimized LayerNorm kernel
__global__ void layer_norm_kernel(
    const float* input,    // [batch_size, seq_len, hidden_size]
    const float* gamma,    // [hidden_size]
    const float* beta,     // [hidden_size]
    float* output,         // [batch_size, seq_len, hidden_size]
    const int batch_size,
    const int seq_len, 
    const int hidden_size,
    const float eps
) {
    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    extern __shared__ float shmem[];
    float* input_cache = shmem;
    
    // Load input into shared memory
    int base_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        input_cache[i] = input[base_idx + i];
    }
    
    __syncthreads();
    
    // Compute mean
    float sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        sum += input_cache[i];
    }
    
    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Block-level reduction
    __shared__ float block_sum[32];
    if (tid % warpSize == 0) {
        block_sum[tid / warpSize] = sum;
    }
    __syncthreads();
    
    if (tid < 32) {
        sum = (tid < blockDim.x / warpSize) ? block_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }
    
    __shared__ float mean;
    if (tid == 0) {
        mean = sum / hidden_size;
    }
    __syncthreads();
    
    // Compute variance
    float var_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float diff = input_cache[i] - mean;
        var_sum += diff * diff;
    }
    
    // Reduction for variance (similar to mean)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
    }
    
    __shared__ float block_var_sum[32];
    if (tid % warpSize == 0) {
        block_var_sum[tid / warpSize] = var_sum;
    }
    __syncthreads();
    
    if (tid < 32) {
        var_sum = (tid < blockDim.x / warpSize) ? block_var_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
        }
    }
    
    __shared__ float inv_std;
    if (tid == 0) {
        inv_std = rsqrtf(var_sum / hidden_size + eps);
    }
    __syncthreads();
    
    // Apply normalization
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (input_cache[i] - mean) * inv_std;
        output[base_idx + i] = normalized * gamma[i] + beta[i];
    }
}
void launch_layer_norm_kernels(
    const float* input, const float* gamma, const float* beta, float* output,
    int batch_size, int seq_len, int hidden_size, float eps,
    cudaStream_t stream
) {
    dim3 block(256);
    dim3 grid(seq_len, batch_size);
    
    size_t shmem_size = hidden_size * sizeof(float);
    
    layer_norm_kernel<<<grid, block, shmem_size, stream>>>(
        input, gamma, beta, output, batch_size, seq_len, hidden_size, eps
    );
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw CudaException("LayerNorm kernel launch failed: " + 
                           std::string(cudaGetErrorString(error)));
    }
}
// INT8 Quantization kernels for optimization
__global__ void quantize_int8_kernel(
    const float* input,
    int8_t* output,
    float* scale,
    const int size
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Find scale (could be precomputed)
    __shared__ float max_val;
    if (threadIdx.x == 0) {
        max_val = 0.0f;
        for (int i = 0; i < size; ++i) {
            max_val = fmaxf(max_val, fabsf(input[i]));
        }
        *scale = max_val / 127.0f;
    }
    __syncthreads();
    
    // Quantize
    float scaled = input[idx] / (*scale);
    output[idx] = static_cast<int8_t>(fmaxf(-128.0f, fminf(127.0f, roundf(scaled))));
}
} // namespace openinferencev2
/**
 * Custom CUDA Kernels Header
 * High-performance kernel declarations for OpenInferencev2
 */
#pragma once

#include <cuda_runtime.h>
#include <string>
#include <stdexcept>

namespace openinferencev2 {

// Exception class for CUDA errors
class CudaException : public std::exception {
public:
    explicit CudaException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

// FlashAttention kernel
void flash_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int seq_len, int num_heads, int head_dim,
    cudaStream_t stream
);

// Fused Feed-Forward Network kernel
void fused_ffn_forward(
    const float* input, const float* gate_weight, const float* up_weight,
    const float* down_weight, float* output,
    int batch_size, int seq_len, int hidden_size, int intermediate_size,
    cudaStream_t stream
);

// Layer normalization kernel
void launch_layer_norm_kernels(
    const float* input, const float* gamma, const float* beta, float* output,
    int batch_size, int seq_len, int hidden_size, float eps,
    cudaStream_t stream
);

// INT8 quantization kernel
void quantize_int8_kernel_launch(
    const float* input, int8_t* output, float* scale, int size,
    cudaStream_t stream
);

} // namespace openinferencev2 
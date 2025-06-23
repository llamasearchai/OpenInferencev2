/**
 * OpenInferencev2 Core Inference Engine Header
 * High-performance C++/CUDA implementation
 */
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <future>

namespace openinferencev2 {

// Forward declarations
class GPUManager;
class KVCache;
class DistributedManager;

// Configuration structure
struct Config {
    int num_gpus = 1;
    int max_batch_size = 32;
    int max_sequence_length = 2048;
    int tensor_parallel_size = 1;
    int pipeline_parallel_size = 1;
    int moe_parallel_size = 1;
    float kv_cache_size_gb = 8.0f;
    bool use_fp16 = true;
    bool use_flash_attention = true;
    bool use_cuda_graphs = true;
    bool use_tensorrt = false;
    bool use_int8_quantization = false;
    std::string distributed_backend = "nccl";
};

// Inference request structure
struct InferenceRequest {
    std::string id;
    std::vector<int> input_tokens;
    int max_tokens = 100;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 50;
    float repetition_penalty = 1.0f;
    std::vector<std::string> stop_sequences;
};

// Inference result structure
struct InferenceResult {
    std::string request_id;
    std::vector<int> output_tokens;
    float latency = 0.0f;
    float tokens_per_second = 0.0f;
    std::string finish_reason = "length";
    bool success = true;
    std::string error_message;
};

// Performance statistics
struct PerformanceStats {
    double avg_latency_ms = 0.0;
    size_t total_inferences = 0;
    std::vector<float> gpu_utilization;
    std::vector<float> gpu_memory_usage;
    float kv_cache_hit_rate = 0.0f;
    float kv_cache_memory_usage = 0.0f;
};

// Tensor structures for internal use
struct InputTensors {
    void* query_tensor = nullptr;
    void* key_tensor = nullptr;
    void* value_tensor = nullptr;
    void* hidden_states = nullptr;
    int batch_size = 0;
    int sequence_length = 0;
};

struct OutputTensors {
    void* attention_output = nullptr;
    void* ffn_output = nullptr;
    int batch_size = 0;
    int sequence_length = 0;
};

// FFN weights structure
struct FFNWeights {
    void* gate_weight = nullptr;
    void* up_weight = nullptr;
    void* down_weight = nullptr;
};

// Exception class for CUDA errors
class CudaException : public std::exception {
public:
    explicit CudaException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

// Main inference engine class
class InferenceEngine {
public:
    explicit InferenceEngine(const Config& config);
    ~InferenceEngine();
    
    // Core functionality
    bool initialize(const std::string& model_path);
    std::vector<InferenceResult> inference(const std::vector<InferenceRequest>& requests);
    PerformanceStats get_performance_stats() const;
    void shutdown();
    
    // Model management
    bool load_model_weights(const std::string& model_path);
    bool initialize_custom_kernels();
    void warmup();
    
    // Batch processing
    std::vector<InferenceResult> process_batch(const std::vector<InferenceRequest>& batch);
    std::vector<std::vector<InferenceRequest>> batch_requests(const std::vector<InferenceRequest>& requests);
    
    // Tensor operations
    InputTensors prepare_input_tensors(const std::vector<InferenceRequest>& batch);
    OutputTensors allocate_output_tensors(int batch_size);
    void process_outputs(const OutputTensors& outputs, 
                        const std::vector<InferenceRequest>& batch,
                        std::vector<InferenceResult>& results);
    
    // CUDA graph optimization
    bool setup_cuda_graphs();
    void execute_cuda_graph(const InputTensors& inputs, const OutputTensors& outputs, void* stream);
    void execute_inference_kernels(const InputTensors& inputs, const OutputTensors& outputs, void* stream);
    void update_graph_parameters(const InputTensors& inputs, const OutputTensors& outputs);
    
    // Kernel launches
    void launch_flash_attention_kernel(const InputTensors& inputs, const OutputTensors& outputs, void* stream);
    void launch_standard_attention_kernel(const InputTensors& inputs, const OutputTensors& outputs, void* stream);
    void launch_ffn_kernels(const InputTensors& inputs, const OutputTensors& outputs, void* stream);
    void launch_layer_norm_kernels(const InputTensors& inputs, const OutputTensors& outputs, void* stream);
    void launch_output_projection_kernel(const InputTensors& inputs, const OutputTensors& outputs, void* stream);
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl;
    
    // Model parameters
    int max_batch_size = 32;
    FFNWeights ffn_weights;
};

} // namespace openinferencev2 
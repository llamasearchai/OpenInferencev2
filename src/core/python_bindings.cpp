/**
 * Python bindings for OpenInferencev2 C++ engine
 * Uses pybind11 for seamless Python integration
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>
#include "inference_engine.h"

namespace py = pybind11;
using namespace openinferencev2;

PYBIND11_MODULE(openinferencev2_cpp, m) {
    m.doc() = "OpenInferencev2 C++ inference engine bindings";
    
    // InferenceRequest structure
    py::class_<InferenceRequest>(m, "InferenceRequest")
        .def(py::init<>())
        .def_readwrite("id", &InferenceRequest::id)
        .def_readwrite("input_tokens", &InferenceRequest::input_tokens)
        .def_readwrite("max_tokens", &InferenceRequest::max_tokens)
        .def_readwrite("temperature", &InferenceRequest::temperature)
        .def_readwrite("top_p", &InferenceRequest::top_p)
        .def_readwrite("top_k", &InferenceRequest::top_k)
        .def_readwrite("repetition_penalty", &InferenceRequest::repetition_penalty)
        .def_readwrite("stop_sequences", &InferenceRequest::stop_sequences);
    
    // InferenceResult structure
    py::class_<InferenceResult>(m, "InferenceResult")
        .def(py::init<>())
        .def_readwrite("request_id", &InferenceResult::request_id)
        .def_readwrite("output_tokens", &InferenceResult::output_tokens)
        .def_readwrite("latency", &InferenceResult::latency)
        .def_readwrite("tokens_per_second", &InferenceResult::tokens_per_second)
        .def_readwrite("finish_reason", &InferenceResult::finish_reason)
        .def_readwrite("success", &InferenceResult::success)
        .def_readwrite("error_message", &InferenceResult::error_message);
    
    // Config structure
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("num_gpus", &Config::num_gpus)
        .def_readwrite("max_batch_size", &Config::max_batch_size)
        .def_readwrite("max_sequence_length", &Config::max_sequence_length)
        .def_readwrite("tensor_parallel_size", &Config::tensor_parallel_size)
        .def_readwrite("pipeline_parallel_size", &Config::pipeline_parallel_size)
        .def_readwrite("use_fp16", &Config::use_fp16)
        .def_readwrite("use_flash_attention", &Config::use_flash_attention)
        .def_readwrite("use_cuda_graphs", &Config::use_cuda_graphs)
        .def_readwrite("kv_cache_size_gb", &Config::kv_cache_size_gb);
    
    // PerformanceStats structure
    py::class_<PerformanceStats>(m, "PerformanceStats")
        .def(py::init<>())
        .def_readwrite("avg_latency_ms", &PerformanceStats::avg_latency_ms)
        .def_readwrite("total_inferences", &PerformanceStats::total_inferences)
        .def_readwrite("gpu_utilization", &PerformanceStats::gpu_utilization)
        .def_readwrite("gpu_memory_usage", &PerformanceStats::gpu_memory_usage)
        .def_readwrite("kv_cache_hit_rate", &PerformanceStats::kv_cache_hit_rate)
        .def_readwrite("kv_cache_memory_usage", &PerformanceStats::kv_cache_memory_usage);
    
    // Main InferenceEngine class
    py::class_<InferenceEngine>(m, "InferenceEngine")
        .def(py::init<const Config&>())
        .def("initialize", &InferenceEngine::initialize,
             "Initialize the inference engine with model path")
        .def("inference", &InferenceEngine::inference,
             "Run inference on a batch of requests")
        .def("get_performance_stats", &InferenceEngine::get_performance_stats,
             "Get current performance statistics")
        .def("shutdown", &InferenceEngine::shutdown,
             "Shutdown the inference engine");
    
    // Utility functions
    m.def("get_cuda_device_count", []() {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        return device_count;
    }, "Get number of available CUDA devices");
    
    m.def("get_cuda_memory_info", [](int device_id) {
        cudaSetDevice(device_id);
        size_t free_memory, total_memory;
        cudaMemGetInfo(&free_memory, &total_memory);
        return std::make_tuple(free_memory, total_memory);
    }, "Get CUDA memory information for device");
} 
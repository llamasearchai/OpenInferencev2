# OpenInferencev2 API Reference

## Core Classes

### OpenInferencev2Engine

The main inference engine class that provides high-performance LLM inference capabilities.

```python
from openinferencev2 import OpenInferencev2Engine, Config

# Initialize engine
config = Config({
    'num_gpus': 2,
    'max_batch_size': 32,
    'use_fp16': True,
    'use_flash_attention': True
})
engine = OpenInferencev2Engine("/path/to/model", config)
```

#### Methods

##### `__init__(model_path: str, config: Config, monitor: Optional[PerformanceMonitor] = None)`
Initialize the OpenInferencev2 engine.

**Parameters:**
- `model_path` (str): Path to the model directory
- `config` (Config): Configuration object with engine settings
- `monitor` (PerformanceMonitor, optional): Performance monitoring instance

##### `async load_model()`
Load and initialize the model asynchronously.

**Returns:** None

**Raises:**
- `RuntimeError`: If model loading fails
- `FileNotFoundError`: If model path doesn't exist

##### `async generate(request: InferenceRequest) -> InferenceResponse`
Generate text for a single inference request.

**Parameters:**
- `request` (InferenceRequest): Input request with prompt and parameters

**Returns:** InferenceResponse with generated text and metadata

##### `async generate_batch(requests: List[InferenceRequest]) -> List[InferenceResponse]`
Process multiple inference requests in a batch.

**Parameters:**
- `requests` (List[InferenceRequest]): List of input requests

**Returns:** List of InferenceResponse objects

##### `async generate_stream(request: InferenceRequest) -> AsyncGenerator[str, None]`
Generate text with streaming output.

**Parameters:**
- `request` (InferenceRequest): Input request

**Yields:** Generated text tokens as they are produced

##### `get_performance_stats() -> Dict`
Get current performance statistics.

**Returns:** Dictionary with performance metrics

##### `async health_check() -> Dict`
Perform health check of the engine.

**Returns:** Dictionary with health status information

### InferenceRequest

Data structure for inference requests.

```python
from openinferencev2 import InferenceRequest

request = InferenceRequest(
    id="unique_request_id",
    prompt="Your input text here",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    repetition_penalty=1.0,
    stop_sequences=["<|endoftext|>"]
)
```

#### Attributes

- `id` (str): Unique identifier for the request
- `prompt` (str): Input text prompt
- `max_tokens` (int, default=100): Maximum number of tokens to generate
- `temperature` (float, default=0.7): Sampling temperature
- `top_p` (float, default=0.9): Top-p sampling parameter
- `top_k` (int, default=50): Top-k sampling parameter
- `repetition_penalty` (float, default=1.0): Repetition penalty factor
- `stop_sequences` (List[str], default=[]): List of stop sequences

### InferenceResponse

Data structure for inference responses.

#### Attributes

- `id` (str): Request identifier
- `text` (str): Generated text
- `tokens` (List[int]): Generated token IDs
- `latency` (float): Inference latency in seconds
- `tokens_per_second` (float): Generation throughput
- `finish_reason` (str): Reason for completion
- `success` (bool): Whether the request succeeded
- `error_message` (str): Error message if failed

### Config

Configuration management class.

```python
from openinferencev2 import Config

config = Config({
    'num_gpus': 2,
    'max_batch_size': 32,
    'use_fp16': True,
    'use_flash_attention': True,
    'tensor_parallel_size': 2,
    'pipeline_parallel_size': 1,
    'kv_cache_size_gb': 16.0,
    'quantization': 'int8'
})
```

#### Key Configuration Options

##### GPU Settings
- `num_gpus` (int): Number of GPUs to use
- `gpu_memory_fraction` (float): Fraction of GPU memory to use
- `tensor_parallel_size` (int): Tensor parallelism degree
- `pipeline_parallel_size` (int): Pipeline parallelism degree

##### Performance Settings
- `max_batch_size` (int): Maximum batch size
- `max_sequence_length` (int): Maximum sequence length
- `use_fp16` (bool): Enable FP16 precision
- `use_flash_attention` (bool): Enable FlashAttention
- `use_cuda_graphs` (bool): Enable CUDA graphs
- `quantization` (str): Quantization method ('int8', 'int4', None)

##### Memory Settings
- `kv_cache_size_gb` (float): KV cache size in GB
- `max_memory_per_gpu_gb` (float): Maximum memory per GPU

### PerformanceMonitor

Real-time performance monitoring and alerting.

```python
from openinferencev2 import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

# Get current metrics
metrics = monitor.get_metrics()
print(f"GPU Usage: {metrics['gpu_usage']}%")
print(f"Memory Usage: {metrics['memory_usage']}%")
```

#### Methods

##### `start()`
Start performance monitoring.

##### `stop()`
Stop performance monitoring.

##### `get_metrics() -> Dict`
Get current performance metrics.

##### `async health_check() -> Dict`
Perform system health check.

##### `record_inference_metrics(latency: float, tokens: int, success: bool)`
Record metrics for an inference request.

### RequestScheduler

Advanced request scheduling and batching.

```python
from openinferencev2 import RequestScheduler

scheduler = RequestScheduler(
    max_batch_size=32,
    max_wait_time=0.1,
    priority_levels=3
)
```

#### Methods

##### `add_request(request: InferenceRequest, priority: int = 0)`
Add a request to the scheduler queue.

##### `async get_batch() -> List[InferenceRequest]`
Get the next batch of requests to process.

##### `get_stats() -> Dict`
Get scheduler statistics.

### ModelOptimizer

Model optimization utilities.

```python
from openinferencev2 import ModelOptimizer

optimizer = ModelOptimizer()
optimizer.optimize_model(model, config)
```

#### Methods

##### `optimize_model(model, config: Config)`
Apply optimizations to a model.

##### `quantize_model(model, method: str)`
Apply quantization to a model.

##### `compile_model(model)`
Compile model with torch.compile.

## CLI Interface

### Command Line Usage

```bash
# Start inference server
python -m src.cli.main --model /path/to/model

# With configuration file
python -m src.cli.main --model /path/to/model --config config.json

# Set log level
python -m src.cli.main --model /path/to/model --log-level DEBUG
```

### CLI Options

- `--model MODEL`: Path to model directory (required)
- `--config CONFIG`: Path to configuration file
- `--log-level LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Exception Handling

### Common Exceptions

#### `ModelLoadError`
Raised when model loading fails.

#### `InferenceError`
Raised when inference fails.

#### `ConfigurationError`
Raised when configuration is invalid.

#### `ResourceError`
Raised when system resources are insufficient.

## Performance Optimization Guidelines

### GPU Memory Management

1. **Monitor Memory Usage**: Use `PerformanceMonitor` to track GPU memory
2. **Optimize Batch Size**: Find optimal batch size for your hardware
3. **KV Cache Management**: Configure appropriate KV cache size
4. **Model Quantization**: Use INT8/INT4 quantization for memory efficiency

### Throughput Optimization

1. **Dynamic Batching**: Enable dynamic batching for better throughput
2. **CUDA Graphs**: Enable CUDA graphs for repetitive workloads
3. **FlashAttention**: Use FlashAttention for memory-efficient attention
4. **Mixed Precision**: Enable FP16 for better performance

### Latency Optimization

1. **Reduce Batch Size**: Smaller batches for lower latency
2. **Pipeline Parallelism**: Use pipeline parallelism for large models
3. **Speculative Decoding**: Enable speculative decoding when available
4. **Preemptive Scheduling**: Use priority-based scheduling

## Examples

### Basic Inference

```python
import asyncio
from openinferencev2 import OpenInferencev2Engine, InferenceRequest, Config

async def main():
    # Configure engine
    config = Config({
        'num_gpus': 1,
        'max_batch_size': 8,
        'use_fp16': True
    })
    
    # Initialize engine
    engine = OpenInferencev2Engine("/path/to/model", config)
    await engine.load_model()
    
    # Create request
    request = InferenceRequest(
        id="example_001",
        prompt="What is artificial intelligence?",
        max_tokens=100,
        temperature=0.7
    )
    
    # Generate response
    response = await engine.generate(request)
    print(f"Generated: {response.text}")
    print(f"Latency: {response.latency:.3f}s")
    print(f"Throughput: {response.tokens_per_second:.1f} tokens/s")

asyncio.run(main())
```

### Batch Processing

```python
import asyncio
from openinferencev2 import OpenInferencev2Engine, InferenceRequest, Config

async def batch_inference():
    config = Config({'num_gpus': 1, 'max_batch_size': 16})
    engine = OpenInferencev2Engine("/path/to/model", config)
    await engine.load_model()
    
    # Create batch of requests
    requests = [
        InferenceRequest(
            id=f"batch_{i}",
            prompt=f"Question {i}: What is machine learning?",
            max_tokens=50
        )
        for i in range(8)
    ]
    
    # Process batch
    responses = await engine.generate_batch(requests)
    
    for response in responses:
        print(f"Request {response.id}: {response.text[:50]}...")

asyncio.run(batch_inference())
```

### Streaming Generation

```python
import asyncio
from openinferencev2 import OpenInferencev2Engine, InferenceRequest, Config

async def streaming_example():
    config = Config({'num_gpus': 1})
    engine = OpenInferencev2Engine("/path/to/model", config)
    await engine.load_model()
    
    request = InferenceRequest(
        id="stream_001",
        prompt="Write a story about artificial intelligence",
        max_tokens=200
    )
    
    print("Streaming output:")
    async for token in engine.generate_stream(request):
        print(token, end='', flush=True)
    print()

asyncio.run(streaming_example())
```

## Best Practices

### Error Handling

```python
import asyncio
from openinferencev2 import OpenInferencev2Engine, InferenceRequest, Config
from openinferencev2.exceptions import ModelLoadError, InferenceError

async def robust_inference():
    try:
        config = Config({'num_gpus': 1})
        engine = OpenInferencev2Engine("/path/to/model", config)
        await engine.load_model()
        
        request = InferenceRequest(
            id="robust_001",
            prompt="Test prompt",
            max_tokens=50
        )
        
        response = await engine.generate(request)
        if response.success:
            print(f"Success: {response.text}")
        else:
            print(f"Inference failed: {response.error_message}")
            
    except ModelLoadError as e:
        print(f"Model loading failed: {e}")
    except InferenceError as e:
        print(f"Inference error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

asyncio.run(robust_inference())
```

### Performance Monitoring

```python
from openinferencev2 import PerformanceMonitor

# Initialize monitor
monitor = PerformanceMonitor()
monitor.start()

# Monitor during inference
metrics = monitor.get_metrics()
if metrics['gpu_usage'] > 95:
    print("Warning: High GPU usage detected")

if metrics['memory_usage'] > 90:
    print("Warning: High memory usage detected")

# Stop monitoring
monitor.stop()
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or sequence length
2. **Model Loading Fails**: Check model path and permissions
3. **Low Performance**: Enable optimizations (FP16, FlashAttention, CUDA graphs)
4. **High Latency**: Reduce batch size or use pipeline parallelism

### Debug Information

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check engine health
health = await engine.health_check()
print(f"Engine health: {health}")

# Get performance stats
stats = engine.get_performance_stats()
print(f"Performance: {stats}")
``` 
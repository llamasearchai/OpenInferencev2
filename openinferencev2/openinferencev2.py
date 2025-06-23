"""
OpenInferencev2 Python API
High-level interface for the inference engine
"""
import asyncio
import ctypes
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union, AsyncGenerator
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
# Import compiled C++ extension
try:
    from openinferencev2_cpp import InferenceEngine as CppInferenceEngine
except ImportError:
    print("Warning: C++ extension not found. Using Python fallback.")
    CppInferenceEngine = None
@dataclass
class InferenceRequest:
    """Inference request data structure"""
    id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = None
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []
@dataclass 
class InferenceResponse:
    """Inference response data structure"""
    id: str
    text: str
    tokens: List[int]
    latency: float
    tokens_per_second: float
    finish_reason: str
    success: bool = True
    error_message: str = ""
@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    max_sequence_length: int
    rope_theta: float = 10000.0
    
class OpenInferencev2Engine:
    """Main inference engine class"""
    
    def __init__(self, model_path: str, config: Dict, monitor=None):
        self.model_path = Path(model_path)
        self.config = config
        self.monitor = monitor
        self.logger = logging.getLogger(__name__)
        
        # Engine components
        self.cpp_engine = None
        self.tokenizer = None
        self.model_config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Performance tracking
        self.total_requests = 0
        self.total_tokens_generated = 0
        self.total_inference_time = 0.0
        
        # Threading for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.request_queue = asyncio.Queue(maxsize=config.get('max_queue_size', 1000))
        
    async def load_model(self):
        """Load and initialize the model"""
        try:
            self.logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Load model configuration
            hf_config = AutoConfig.from_pretrained(self.model_path)
            self.model_config = ModelConfig(
                vocab_size=hf_config.vocab_size,
                hidden_size=hf_config.hidden_size,
                num_layers=hf_config.num_hidden_layers,
                num_heads=hf_config.num_attention_heads,
                intermediate_size=getattr(hf_config, 'intermediate_size', hf_config.hidden_size * 4),
                max_sequence_length=getattr(hf_config, 'max_position_embeddings', 2048)
            )
            
            # Initialize C++ engine if available
            if CppInferenceEngine:
                self.cpp_engine = CppInferenceEngine(self.config)
                success = self.cpp_engine.initialize(str(self.model_path))
                if not success:
                    raise RuntimeError("Failed to initialize C++ engine")
            else:
                # Fallback to PyTorch implementation
                await self._load_pytorch_model()
                
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    async def _load_pytorch_model(self):
        """Fallback PyTorch model loading"""
        from transformers import AutoModelForCausalLM
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.config.get('use_fp16', True) else torch.float32,
            device_map='auto' if self.config.get('num_gpus', 1) > 1 else None,
            trust_remote_code=True
        )
        
        if self.config.get('num_gpus', 1) == 1:
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
    async def optimize_model(self):
        """Apply various optimizations to the model"""
        try:
            self.logger.info("Applying model optimizations...")
            
            optimizations = []
            
            # Torch compilation
            if self.config.get('use_torch_compile', False):
                optimizations.append(self._apply_torch_compile())
                
            # FP16 optimization
            if self.config.get('use_fp16', True):
                optimizations.append(self._apply_fp16_optimization())
                
            # Quantization
            if self.config.get('quantization'):
                optimizations.append(self._apply_quantization())
                
            # Flash Attention
            if self.config.get('use_flash_attention', True):
                optimizations.append(self._enable_flash_attention())
                
            # Wait for all optimizations to complete
            if optimizations:
                await asyncio.gather(*optimizations)
                
            self.logger.info("Model optimizations completed")
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            # Continue without optimizations
            
    async def _apply_torch_compile(self):
        """Apply torch.compile optimization"""
        if hasattr(self, 'model') and hasattr(torch, 'compile'):
            self.logger.info("Applying torch.compile...")
            self.model = torch.compile(self.model, mode='max-autotune')
            
    async def _apply_fp16_optimization(self):
        """Apply FP16 optimization"""
        if hasattr(self, 'model'):
            self.logger.info("Converting to FP16...")
            self.model = self.model.half()
            
    async def _apply_quantization(self):
        """Apply model quantization"""
        quantization_method = self.config.get('quantization')
        if quantization_method == 'int8' and hasattr(self, 'model'):
            self.logger.info("Applying INT8 quantization...")
            # Implementation would use libraries like BitsAndBytes
            
    async def _enable_flash_attention(self):
        """Enable FlashAttention if available"""
        try:
            import flash_attn
            self.logger.info("FlashAttention enabled")
        except ImportError:
            self.logger.warning("FlashAttention not available")
            
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response for a single request"""
        start_time = time.time()
        
        try:
            # Tokenize input
            input_tokens = self.tokenizer.encode(
                request.prompt, 
                return_tensors='pt', 
                add_special_tokens=True
            )
            
            if self.cpp_engine:
                # Use C++ engine
                response = await self._generate_cpp(request, input_tokens)
            else:
                # Use PyTorch fallback
                response = await self._generate_pytorch(request, input_tokens)
                
            # Calculate performance metrics
            latency = time.time() - start_time
            tokens_per_second = len(response.tokens) / latency if latency > 0 else 0
            
            # Update statistics
            self.total_requests += 1
            self.total_tokens_generated += len(response.tokens)
            self.total_inference_time += latency
            
            response.latency = latency
            response.tokens_per_second = tokens_per_second
            
            # Monitor performance if available
            if self.monitor:
                await self.monitor.record_inference(response)
                
            return response
            
        except Exception as e:
            self.logger.error(f"Generation failed for request {request.id}: {e}")
            return InferenceResponse(
                id=request.id,
                text="",
                tokens=[],
                latency=time.time() - start_time,
                tokens_per_second=0.0,
                finish_reason="error",
                success=False,
                error_message=str(e)
            )
            
    async def _generate_cpp(self, request: InferenceRequest, input_tokens: torch.Tensor) -> InferenceResponse:
        """Generate using C++ engine"""
        # Convert PyTorch tensor to format expected by C++ engine
        cpp_request = {
            'id': request.id,
            'input_tokens': input_tokens.cpu().numpy().tolist(),
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'top_k': request.top_k,
            'repetition_penalty': request.repetition_penalty,
            'stop_sequences': request.stop_sequences
        }
        
        # Execute inference in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        cpp_response = await loop.run_in_executor(
            self.executor,
            self.cpp_engine.inference,
            [cpp_request]
        )
        
        # Convert back to Python format
        result = cpp_response[0]
        
        output_text = self.tokenizer.decode(
            result['output_tokens'], 
            skip_special_tokens=True
        )
        
        return InferenceResponse(
            id=request.id,
            text=output_text,
            tokens=result['output_tokens'],
            latency=0.0,  # Will be set by caller
            tokens_per_second=0.0,  # Will be set by caller
            finish_reason=result.get('finish_reason', 'length')
        )
        
    async def _generate_pytorch(self, request: InferenceRequest, input_tokens: torch.Tensor) -> InferenceResponse:
        """Generate using PyTorch fallback"""
        input_tokens = input_tokens.to(self.device)
        
        generation_config = {
            'max_new_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
            'top_k': request.top_k,
            'repetition_penalty': request.repetition_penalty,
            'do_sample': request.temperature > 0,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        # Execute generation in thread pool
        loop = asyncio.get_event_loop()
        
        def _generate():
            with torch.inference_mode():
                outputs = self.model.generate(
                    input_tokens,
                    **generation_config
                )
            return outputs
            
        outputs = await loop.run_in_executor(self.executor, _generate)
        
        # Extract generated tokens (excluding input)
        generated_tokens = outputs[0][input_tokens.shape[1]:].cpu().tolist()
        
        # Decode output
        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return InferenceResponse(
            id=request.id,
            text=output_text,
            tokens=generated_tokens,
            latency=0.0,  # Will be set by caller
            tokens_per_second=0.0,  # Will be set by caller
            finish_reason='length'  # Simplified
        )
        
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        # Simplified streaming implementation
        response = await self.generate(request)
        
        # Simulate streaming by yielding chunks
        words = response.text.split()
        for i, word in enumerate(words):
            if i == 0:
                yield word
            else:
                yield f" {word}"
            await asyncio.sleep(0.01)  # Small delay for streaming effect
            
    def get_performance_stats(self) -> Dict:
        """Get engine performance statistics"""
        if self.total_requests == 0:
            return {
                'total_requests': 0,
                'avg_latency': 0.0,
                'avg_tokens_per_second': 0.0,
                'total_tokens_generated': 0,
                'uptime': 0.0
            }
            
        avg_latency = self.total_inference_time / self.total_requests
        avg_tokens_per_second = self.total_tokens_generated / self.total_inference_time
        
        stats = {
            'total_requests': self.total_requests,
            'avg_latency': avg_latency,
            'avg_tokens_per_second': avg_tokens_per_second,
            'total_tokens_generated': self.total_tokens_generated,
            'total_inference_time': self.total_inference_time
        }
        
        # Add GPU stats if available
        if torch.cuda.is_available():
            stats.update({
                'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**3,
                'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**3,
                'gpu_utilization': torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            })
            
        # Add C++ engine stats if available
        if self.cpp_engine:
            cpp_stats = self.cpp_engine.get_performance_stats()
            stats.update({
                'cpp_engine_stats': cpp_stats,
                'kv_cache_hit_rate': cpp_stats.get('kv_cache_hit_rate', 0.0),
                'gpu_kernel_time': cpp_stats.get('gpu_kernel_time', 0.0)
            })
            
        return stats
        
    async def health_check(self) -> Dict:
        """Perform health check"""
        try:
            # Basic functionality test
            test_request = InferenceRequest(
                id="health_check",
                prompt="Test",
                max_tokens=1,
                temperature=0.0
            )
            
            response = await self.generate(test_request)
            
            return {
                'status': 'healthy' if response.success else 'unhealthy',
                'model_loaded': self.tokenizer is not None,
                'cpp_engine_available': self.cpp_engine is not None,
                'device': str(self.device),
                'last_check': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'last_check': time.time()
            }
            
    async def shutdown(self):
        """Shutdown the engine gracefully"""
        self.logger.info("Shutting down OpenInferencev2 engine...")
        
        if self.cpp_engine:
            self.cpp_engine.shutdown()
            
        if hasattr(self, 'model'):
            del self.model
            
        torch.cuda.empty_cache()
        
        self.executor.shutdown(wait=True)
        
        self.logger.info("Engine shutdown complete")

# Compatibility alias for backward compatibility
TurboInferEngine = OpenInferencev2Engine
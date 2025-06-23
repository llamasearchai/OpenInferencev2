"""
Model optimization utilities for OpenInferencev2
Advanced optimization techniques for high-performance inference
"""
import torch
import logging
import asyncio
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
import tempfile
import json

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Advanced model optimization utilities for OpenInferencev2"""
    
    def __init__(self, engine):
        self.engine = engine
        self.logger = logging.getLogger(self.__class__.__name__)
        self.optimization_history = []
        self.current_optimizations = set()
        
    async def optimize_all(self) -> Dict[str, Any]:
        """Apply all available optimizations"""
        results = {}
        optimizations = [
            ('fp16', self.convert_to_fp16),
            ('torch_compile', self.apply_torch_compile),
            ('flash_attention', self.enable_flash_attention),
            ('cuda_graphs', self.enable_cuda_graphs),
            ('kv_cache', self.optimize_kv_cache),
        ]
        
        for name, optimization_func in optimizations:
            try:
                result = await optimization_func()
                results[name] = {
                    'success': True,
                    'result': result,
                    'timestamp': time.time()
                }
                self.current_optimizations.add(name)
            except Exception as e:
                self.logger.error(f"Optimization {name} failed: {e}")
                results[name] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': time.time()
                }
                
        self.optimization_history.append(results)
        return results
        
    async def convert_to_fp16(self) -> Dict[str, Any]:
        """Convert model to FP16 precision"""
        if not hasattr(self.engine, 'model'):
            return {'status': 'skipped', 'reason': 'No model available'}
            
        try:
            self.logger.info("Converting model to FP16...")
            start_time = time.time()
            
            # Get initial memory usage
            initial_memory = self._get_memory_usage()
            
            # Convert to half precision
            self.engine.model = self.engine.model.half()
            
            # Force garbage collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            conversion_time = time.time() - start_time
            final_memory = self._get_memory_usage()
            memory_saved = initial_memory - final_memory
            
            result = {
                'status': 'completed',
                'conversion_time': conversion_time,
                'memory_saved_mb': memory_saved,
                'memory_reduction_percent': (memory_saved / initial_memory) * 100 if initial_memory > 0 else 0
            }
            
            self.logger.info(f"FP16 conversion completed in {conversion_time:.2f}s, saved {memory_saved:.1f}MB")
            return result
            
        except Exception as e:
            self.logger.error(f"FP16 conversion failed: {e}")
            raise
            
    async def quantize_int8(self) -> Dict[str, Any]:
        """Apply INT8 quantization using torch.quantization"""
        if not hasattr(self.engine, 'model'):
            return {'status': 'skipped', 'reason': 'No model available'}
            
        try:
            self.logger.info("Applying INT8 quantization...")
            start_time = time.time()
            
            # Prepare model for quantization
            self.engine.model.eval()
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.engine.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            
            # Replace model
            original_size = self._get_model_size(self.engine.model)
            self.engine.model = quantized_model
            quantized_size = self._get_model_size(self.engine.model)
            
            quantization_time = time.time() - start_time
            size_reduction = original_size - quantized_size
            
            result = {
                'status': 'completed',
                'quantization_time': quantization_time,
                'original_size_mb': original_size / (1024 * 1024),
                'quantized_size_mb': quantized_size / (1024 * 1024),
                'size_reduction_mb': size_reduction / (1024 * 1024),
                'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0
            }
            
            self.logger.info(f"INT8 quantization completed in {quantization_time:.2f}s, "
                           f"size reduced by {size_reduction/(1024*1024):.1f}MB")
            return result
            
        except Exception as e:
            self.logger.error(f"INT8 quantization failed: {e}")
            raise
            
    async def enable_cuda_graphs(self) -> Dict[str, Any]:
        """Enable CUDA graph optimization"""
        if not torch.cuda.is_available():
            return {'status': 'skipped', 'reason': 'CUDA not available'}
            
        if not hasattr(self.engine, 'model'):
            return {'status': 'skipped', 'reason': 'No model available'}
            
        try:
            self.logger.info("Enabling CUDA graphs...")
            start_time = time.time()
            
            # This is a simplified implementation
            # Real CUDA graphs require capturing the entire forward pass
            self.engine.config['use_cuda_graphs'] = True
            
            setup_time = time.time() - start_time
            
            result = {
                'status': 'completed',
                'setup_time': setup_time,
                'enabled': True
            }
            
            self.logger.info(f"CUDA graphs enabled in {setup_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"CUDA graphs setup failed: {e}")
            raise
            
    async def compile_tensorrt(self) -> Dict[str, Any]:
        """Compile model with TensorRT"""
        try:
            import torch_tensorrt
            available = True
        except ImportError:
            return {'status': 'skipped', 'reason': 'TensorRT not available'}
            
        if not hasattr(self.engine, 'model'):
            return {'status': 'skipped', 'reason': 'No model available'}
            
        try:
            self.logger.info("Compiling with TensorRT...")
            start_time = time.time()
            
            # Prepare model for TensorRT compilation
            self.engine.model.eval()
            
            # Create sample input for compilation
            sample_input = torch.randint(0, 1000, (1, 512), device='cuda')
            
            # Compile with TensorRT
            compiled_model = torch_tensorrt.compile(
                self.engine.model,
                inputs=[sample_input],
                enabled_precisions={torch.float16} if self.engine.config.get('use_fp16') else {torch.float32}
            )
            
            self.engine.model = compiled_model
            
            compilation_time = time.time() - start_time
            
            result = {
                'status': 'completed',
                'compilation_time': compilation_time,
                'precision': 'fp16' if self.engine.config.get('use_fp16') else 'fp32'
            }
            
            self.logger.info(f"TensorRT compilation completed in {compilation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"TensorRT compilation failed: {e}")
            raise
            
    async def apply_torch_compile(self) -> Dict[str, Any]:
        """Apply torch.compile optimization"""
        if not hasattr(torch, 'compile'):
            return {'status': 'skipped', 'reason': 'torch.compile not available'}
            
        if not hasattr(self.engine, 'model'):
            return {'status': 'skipped', 'reason': 'No model available'}
            
        try:
            self.logger.info("Applying torch.compile...")
            start_time = time.time()
            
            # Apply torch.compile with aggressive optimization
            self.engine.model = torch.compile(
                self.engine.model,
                mode='max-autotune',
                dynamic=False
            )
            
            compilation_time = time.time() - start_time
            
            result = {
                'status': 'completed',
                'compilation_time': compilation_time,
                'mode': 'max-autotune'
            }
            
            self.logger.info(f"torch.compile applied in {compilation_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"torch.compile failed: {e}")
            raise
            
    async def enable_flash_attention(self) -> Dict[str, Any]:
        """Enable FlashAttention optimization"""
        try:
            import flash_attn
            available = True
        except ImportError:
            return {'status': 'skipped', 'reason': 'FlashAttention not available'}
            
        try:
            self.logger.info("Enabling FlashAttention...")
            start_time = time.time()
            
            # Enable flash attention in config
            self.engine.config['use_flash_attention'] = True
            
            setup_time = time.time() - start_time
            
            result = {
                'status': 'completed',
                'setup_time': setup_time,
                'version': getattr(flash_attn, '__version__', 'unknown')
            }
            
            self.logger.info(f"FlashAttention enabled in {setup_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"FlashAttention setup failed: {e}")
            raise
            
    async def setup_speculative_decoding(self, draft_model_path: Optional[str] = None) -> Dict[str, Any]:
        """Setup speculative decoding with draft model"""
        if not draft_model_path:
            return {'status': 'skipped', 'reason': 'No draft model path provided'}
            
        try:
            self.logger.info("Setting up speculative decoding...")
            start_time = time.time()
            
            # This would load and configure a smaller draft model
            # For now, we'll just set the configuration
            self.engine.config['use_speculative_decoding'] = True
            self.engine.config['draft_model_path'] = draft_model_path
            
            setup_time = time.time() - start_time
            
            result = {
                'status': 'completed',
                'setup_time': setup_time,
                'draft_model_path': draft_model_path
            }
            
            self.logger.info(f"Speculative decoding setup completed in {setup_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Speculative decoding setup failed: {e}")
            raise
            
    async def optimize_kv_cache(self) -> Dict[str, Any]:
        """Optimize KV cache configuration"""
        try:
            self.logger.info("Optimizing KV cache...")
            start_time = time.time()
            
            # Get current memory info
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory
                current_memory = torch.cuda.memory_allocated(device)
                available_memory = total_memory - current_memory
                
                # Calculate optimal KV cache size (use 50% of available memory)
                optimal_cache_size = (available_memory * 0.5) / (1024 ** 3)  # GB
                
                # Update cache size
                original_size = self.engine.config.get('kv_cache_size_gb', 8.0)
                self.engine.config['kv_cache_size_gb'] = min(optimal_cache_size, 32.0)  # Cap at 32GB
                
                setup_time = time.time() - start_time
                
                result = {
                    'status': 'completed',
                    'setup_time': setup_time,
                    'original_cache_size_gb': original_size,
                    'optimized_cache_size_gb': self.engine.config['kv_cache_size_gb'],
                    'total_gpu_memory_gb': total_memory / (1024 ** 3),
                    'available_memory_gb': available_memory / (1024 ** 3)
                }
            else:
                result = {
                    'status': 'skipped',
                    'reason': 'CUDA not available'
                }
                
            self.logger.info("KV cache optimization completed")
            return result
            
        except Exception as e:
            self.logger.error(f"KV cache optimization failed: {e}")
            raise
            
    async def benchmark_optimizations(self, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark the effect of optimizations"""
        if not test_prompts:
            test_prompts = ["What is artificial intelligence?", "Explain quantum computing.", "Write a Python function."]
            
        try:
            self.logger.info("Benchmarking optimizations...")
            
            # Import here to avoid circular dependency
            from .openinferencev2 import InferenceRequest
            
            benchmark_results = {
                'test_prompts': test_prompts,
                'results': [],
                'optimizations_applied': list(self.current_optimizations)
            }
            
            for i, prompt in enumerate(test_prompts):
                request = InferenceRequest(
                    id=f"benchmark_{i}",
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.0  # Deterministic for benchmarking
                )
                
                start_time = time.time()
                response = await self.engine.generate(request)
                end_time = time.time()
                
                benchmark_results['results'].append({
                    'prompt': prompt,
                    'latency': response.latency,
                    'tokens_per_second': response.tokens_per_second,
                    'tokens_generated': len(response.tokens) if response.tokens else 0,
                    'success': response.success,
                    'total_time': end_time - start_time
                })
                
            # Calculate aggregate metrics
            successful_results = [r for r in benchmark_results['results'] if r['success']]
            
            if successful_results:
                benchmark_results['aggregate'] = {
                    'avg_latency': sum(r['latency'] for r in successful_results) / len(successful_results),
                    'avg_tokens_per_second': sum(r['tokens_per_second'] for r in successful_results) / len(successful_results),
                    'total_tokens': sum(r['tokens_generated'] for r in successful_results),
                    'success_rate': len(successful_results) / len(benchmark_results['results'])
                }
            else:
                benchmark_results['aggregate'] = {
                    'avg_latency': 0,
                    'avg_tokens_per_second': 0,
                    'total_tokens': 0,
                    'success_rate': 0
                }
                
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Benchmarking failed: {e}")
            raise
            
    def _get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
        return 0.0
        
    def _get_model_size(self, model) -> int:
        """Get model size in bytes"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
            
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
            
        return param_size + buffer_size
        
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'applied_optimizations': list(self.current_optimizations),
            'optimization_history': self.optimization_history,
            'available_optimizations': [
                'fp16', 'torch_compile', 'flash_attention', 'cuda_graphs',
                'kv_cache', 'int8_quantization', 'tensorrt', 'speculative_decoding'
            ],
            'last_optimization_time': self.optimization_history[-1]['timestamp'] if self.optimization_history else None
        }
        
    async def reset_optimizations(self) -> Dict[str, Any]:
        """Reset all optimizations (requires model reload)"""
        try:
            self.logger.info("Resetting optimizations...")
            
            # Clear optimization flags
            self.current_optimizations.clear()
            
            # Reset relevant config options
            reset_config = {
                'use_fp16': False,
                'use_flash_attention': False,
                'use_cuda_graphs': False,
                'use_tensorrt': False,
                'use_speculative_decoding': False,
                'quantization': None
            }
            
            self.engine.config.update(reset_config)
            
            return {
                'status': 'completed',
                'message': 'Optimizations reset. Model reload required for full effect.',
                'reset_optimizations': list(self.current_optimizations),
                'timestamp': time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Reset optimizations failed: {e}")
            raise
            
    async def save_optimization_profile(self, profile_path: str) -> Dict[str, Any]:
        """Save current optimization profile to file"""
        try:
            profile_data = {
                'optimizations': list(self.current_optimizations),
                'config': self.engine.config,
                'history': self.optimization_history,
                'timestamp': time.time(),
                'engine_version': getattr(self.engine, '__version__', 'unknown')
            }
            
            with open(profile_path, 'w') as f:
                json.dump(profile_data, f, indent=2)
                
            return {
                'status': 'completed',
                'profile_path': profile_path,
                'optimizations_saved': len(self.current_optimizations)
            }
            
        except Exception as e:
            self.logger.error(f"Save optimization profile failed: {e}")
            raise
            
    async def load_optimization_profile(self, profile_path: str) -> Dict[str, Any]:
        """Load optimization profile from file"""
        try:
            with open(profile_path, 'r') as f:
                profile_data = json.load(f)
                
            # Apply optimizations from profile
            results = {}
            for optimization in profile_data.get('optimizations', []):
                if hasattr(self, f'enable_{optimization}') or hasattr(self, f'apply_{optimization}'):
                    method_name = f'enable_{optimization}' if hasattr(self, f'enable_{optimization}') else f'apply_{optimization}'
                    method = getattr(self, method_name)
                    try:
                        result = await method()
                        results[optimization] = result
                    except Exception as e:
                        results[optimization] = {'status': 'failed', 'error': str(e)}
                        
            return {
                'status': 'completed',
                'profile_path': profile_path,
                'loaded_optimizations': profile_data.get('optimizations', []),
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Load optimization profile failed: {e}")
            raise

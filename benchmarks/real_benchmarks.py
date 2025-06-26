"""
Real-world benchmarking using open-source datasets from Hugging Face
Author: Nik Jois (nikjois@llamasearch.ai)
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

from openinferencev2 import OpenInferencev2Engine, InferenceRequest, Config
from .performance_metrics import PerformanceMetrics, BenchmarkResults

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset benchmarking"""
    name: str
    subset: Optional[str] = None
    split: str = "test"
    max_samples: int = 100
    text_field: str = "text"
    max_length: int = 512


class DatasetBenchmark:
    """Benchmark using real datasets from Hugging Face"""
    
    # Popular open-source datasets for benchmarking
    AVAILABLE_DATASETS = {
        "wikitext": DatasetConfig(
            name="wikitext",
            subset="wikitext-2-raw-v1",
            text_field="text",
            max_samples=50
        ),
        "openwebtext": DatasetConfig(
            name="openwebtext",
            text_field="text",
            max_samples=50
        ),
        "c4": DatasetConfig(
            name="c4",
            subset="en",
            text_field="text",
            max_samples=25  # Smaller sample for C4 as it's large
        ),
        "squad": DatasetConfig(
            name="squad",
            text_field="context",
            max_samples=100
        ),
        "cnn_dailymail": DatasetConfig(
            name="cnn_dailymail",
            subset="3.0.0",
            text_field="article",
            max_samples=50
        )
    }
    
    def __init__(self, engine: OpenInferencev2Engine):
        self.engine = engine
        self.datasets_cache = {}
        
    def load_dataset_samples(self, dataset_config: DatasetConfig) -> List[str]:
        """Load samples from a Hugging Face dataset"""
        if not HF_DATASETS_AVAILABLE:
            logger.warning("Hugging Face datasets not available, using synthetic data")
            return self._generate_synthetic_samples(dataset_config.max_samples)
            
        try:
            cache_key = f"{dataset_config.name}_{dataset_config.subset}_{dataset_config.split}"
            if cache_key in self.datasets_cache:
                return self.datasets_cache[cache_key]
                
            logger.info(f"Loading dataset: {dataset_config.name}")
            
            if dataset_config.subset:
                dataset = load_dataset(
                    dataset_config.name, 
                    dataset_config.subset, 
                    split=dataset_config.split,
                    streaming=True  # Use streaming for large datasets
                )
            else:
                dataset = load_dataset(
                    dataset_config.name,
                    split=dataset_config.split,
                    streaming=True
                )
            
            # Extract text samples
            samples = []
            for i, example in enumerate(dataset):
                if i >= dataset_config.max_samples:
                    break
                    
                text = example.get(dataset_config.text_field, "")
                if isinstance(text, str) and len(text.strip()) > 10:
                    # Truncate to max_length
                    if len(text) > dataset_config.max_length:
                        text = text[:dataset_config.max_length]
                    samples.append(text.strip())
            
            if not samples:
                logger.warning(f"No valid samples found in {dataset_config.name}")
                return self._generate_synthetic_samples(dataset_config.max_samples)
                
            self.datasets_cache[cache_key] = samples
            logger.info(f"Loaded {len(samples)} samples from {dataset_config.name}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_config.name}: {e}")
            return self._generate_synthetic_samples(dataset_config.max_samples)
    
    def _generate_synthetic_samples(self, count: int) -> List[str]:
        """Generate synthetic samples for testing when datasets unavailable"""
        samples = []
        base_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
            "Transformers have revolutionized the field of natural language processing."
        ]
        
        for i in range(count):
            # Create varied length samples
            base = base_texts[i % len(base_texts)]
            multiplier = (i % 5) + 1
            sample = " ".join([base] * multiplier)
            samples.append(sample)
            
        return samples
    
    async def benchmark_dataset(
        self, 
        dataset_name: str, 
        batch_sizes: List[int] = [1, 4, 8],
        max_tokens: int = 100
    ) -> Dict[str, BenchmarkResults]:
        """Benchmark inference on a specific dataset"""
        
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Dataset {dataset_name} not available. Choose from: {list(self.AVAILABLE_DATASETS.keys())}")
        
        dataset_config = self.AVAILABLE_DATASETS[dataset_name]
        samples = self.load_dataset_samples(dataset_config)
        
        if not samples:
            raise RuntimeError(f"No samples loaded from dataset {dataset_name}")
        
        logger.info(f"Benchmarking {dataset_name} with {len(samples)} samples")
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            # Create batches
            batches = []
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i + batch_size]
                batches.append(batch)
            
            # Run benchmark
            batch_results = []
            total_tokens = 0
            total_time = 0
            
            for batch_idx, batch in enumerate(batches):
                if batch_idx >= 10:  # Limit to 10 batches for reasonable test time
                    break
                    
                start_time = time.time()
                
                # Create inference requests
                requests = []
                for text in batch:
                    request = InferenceRequest(
                        id=f"bench_{batch_idx}_{len(requests)}",
                        prompt=text,
                        max_tokens=max_tokens,
                        temperature=0.7
                    )
                    requests.append(request)
                
                # Run inference
                try:
                    if len(requests) == 1:
                        response = await self.engine.generate(requests[0])
                        responses = [response]
                    else:
                        responses = await self.engine.generate_batch(requests)
                    
                    end_time = time.time()
                    batch_time = end_time - start_time
                    
                    # Calculate metrics
                    batch_tokens = sum(len(r.text.split()) for r in responses if r.success)
                    total_tokens += batch_tokens
                    total_time += batch_time
                    
                    batch_results.append({
                        'batch_size': len(requests),
                        'latency': batch_time,
                        'tokens_generated': batch_tokens,
                        'tokens_per_second': batch_tokens / batch_time if batch_time > 0 else 0,
                        'success_rate': sum(1 for r in responses if r.success) / len(responses)
                    })
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    batch_results.append({
                        'batch_size': len(requests),
                        'latency': 0,
                        'tokens_generated': 0,
                        'tokens_per_second': 0,
                        'success_rate': 0
                    })
            
            # Aggregate results
            if batch_results:
                successful_batches = [r for r in batch_results if r['success_rate'] > 0]
                
                if successful_batches:
                    avg_latency = statistics.mean(r['latency'] for r in successful_batches)
                    avg_throughput = statistics.mean(r['tokens_per_second'] for r in successful_batches)
                    total_throughput = total_tokens / total_time if total_time > 0 else 0
                    
                    results[f"batch_{batch_size}"] = BenchmarkResults(
                        dataset_name=dataset_name,
                        batch_size=batch_size,
                        avg_latency=avg_latency,
                        p95_latency=sorted([r['latency'] for r in successful_batches])[int(len(successful_batches) * 0.95)] if len(successful_batches) > 1 else avg_latency,
                        avg_throughput=avg_throughput,
                        total_throughput=total_throughput,
                        success_rate=statistics.mean(r['success_rate'] for r in batch_results),
                        total_samples=len(samples),
                        batches_processed=len(successful_batches)
                    )
        
        return results


class RealWorldBenchmark:
    """Comprehensive real-world benchmarking suite"""
    
    def __init__(self, engine: OpenInferencev2Engine):
        self.engine = engine
        self.dataset_benchmark = DatasetBenchmark(engine)
        
    async def run_comprehensive_benchmark(
        self,
        datasets: List[str] = ["wikitext", "squad"],
        batch_sizes: List[int] = [1, 4, 8],
        max_tokens: int = 100
    ) -> Dict[str, Dict[str, BenchmarkResults]]:
        """Run comprehensive benchmarks across multiple datasets"""
        
        logger.info("Starting comprehensive real-world benchmark")
        logger.info(f"Datasets: {datasets}")
        logger.info(f"Batch sizes: {batch_sizes}")
        
        all_results = {}
        
        for dataset_name in datasets:
            try:
                logger.info(f"Benchmarking dataset: {dataset_name}")
                dataset_results = await self.dataset_benchmark.benchmark_dataset(
                    dataset_name, batch_sizes, max_tokens
                )
                all_results[dataset_name] = dataset_results
                
            except Exception as e:
                logger.error(f"Failed to benchmark {dataset_name}: {e}")
                all_results[dataset_name] = {}
        
        return all_results
    
    def generate_benchmark_report(
        self, 
        results: Dict[str, Dict[str, BenchmarkResults]]
    ) -> str:
        """Generate a comprehensive benchmark report"""
        
        report = []
        report.append("OpenInferencev2 Real-World Benchmark Report")
        report.append("=" * 50)
        report.append(f"Author: Nik Jois (nikjois@llamasearch.ai)")
        report.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        for dataset_name, dataset_results in results.items():
            if not dataset_results:
                continue
                
            report.append(f"Dataset: {dataset_name}")
            report.append("-" * 30)
            
            # Create table
            report.append(f"{'Batch Size':<12} {'Avg Latency (s)':<15} {'P95 Latency (s)':<15} {'Throughput (t/s)':<15} {'Success Rate':<12}")
            report.append("-" * 75)
            
            for config_name, result in dataset_results.items():
                report.append(
                    f"{result.batch_size:<12} "
                    f"{result.avg_latency:<15.3f} "
                    f"{result.p95_latency:<15.3f} "
                    f"{result.avg_throughput:<15.1f} "
                    f"{result.success_rate:<12.2%}"
                )
            
            report.append("")
        
        # Summary
        report.append("Summary")
        report.append("-" * 20)
        
        all_results = []
        for dataset_results in results.values():
            all_results.extend(dataset_results.values())
        
        if all_results:
            avg_latency = statistics.mean(r.avg_latency for r in all_results)
            avg_throughput = statistics.mean(r.avg_throughput for r in all_results)
            avg_success_rate = statistics.mean(r.success_rate for r in all_results)
            
            report.append(f"Overall Average Latency: {avg_latency:.3f}s")
            report.append(f"Overall Average Throughput: {avg_throughput:.1f} tokens/s")
            report.append(f"Overall Success Rate: {avg_success_rate:.2%}")
        
        report.append("")
        report.append("Note: Benchmarks conducted using real open-source datasets from Hugging Face")
        report.append("Performance may vary based on hardware configuration and model size")
        
        return "\n".join(report) 
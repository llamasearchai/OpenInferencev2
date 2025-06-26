#!/usr/bin/env python3
"""
Real-world benchmarking script for OpenInferencev2
Uses actual open-source datasets from Hugging Face for performance testing
Author: Nik Jois (nikjois@llamasearch.ai)
"""

import asyncio
import logging
import time
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from openinferencev2 import OpenInferencev2Engine, Config
from benchmarks import RealWorldBenchmark, DatasetBenchmark, BenchmarkSuite

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_basic_benchmark():
    """Run a basic benchmark with synthetic data"""
    logger.info("Starting basic benchmark with synthetic data")
    
    # Create engine with test configuration
    config = Config({
        'num_gpus': 1,
        'max_batch_size': 8,
        'use_fp16': False,  # Use FP32 for consistency
        'max_sequence_length': 512
    })
    
    engine = OpenInferencev2Engine("test_model", config)
    
    # Initialize benchmark suite
    benchmark = RealWorldBenchmark(engine)
    
    # Run with synthetic data (when HF datasets not available)
    try:
        results = await benchmark.run_comprehensive_benchmark(
            datasets=["wikitext"],  # Will fall back to synthetic if HF not available
            batch_sizes=[1, 2, 4],
            max_tokens=50
        )
        
        # Generate report
        report = benchmark.generate_benchmark_report(results)
        print("\n" + "="*60)
        print("REAL-WORLD BENCHMARK RESULTS")
        print("="*60)
        print(report)
        print("="*60)
        
        # Save results
        suite = BenchmarkSuite()
        for dataset_name, dataset_results in results.items():
            for config_name, result in dataset_results.items():
                suite.add_result(result)
        
        # Save to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_results_{timestamp}.json"
        suite.save_to_file(filename)
        logger.info(f"Benchmark results saved to {filename}")
        
        return suite
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


async def run_dataset_specific_benchmark():
    """Run benchmark on specific dataset configurations"""
    logger.info("Starting dataset-specific benchmark")
    
    config = Config({
        'num_gpus': 1,
        'max_batch_size': 4,
        'use_fp16': False
    })
    
    engine = OpenInferencev2Engine("test_model", config)
    dataset_benchmark = DatasetBenchmark(engine)
    
    # Test available datasets
    available_datasets = list(dataset_benchmark.AVAILABLE_DATASETS.keys())
    logger.info(f"Available datasets: {available_datasets}")
    
    results = {}
    
    for dataset_name in available_datasets[:2]:  # Test first 2 datasets
        try:
            logger.info(f"Testing dataset: {dataset_name}")
            dataset_results = await dataset_benchmark.benchmark_dataset(
                dataset_name,
                batch_sizes=[1, 2],
                max_tokens=25
            )
            results[dataset_name] = dataset_results
            
        except Exception as e:
            logger.error(f"Failed to benchmark {dataset_name}: {e}")
            continue
    
    return results


def analyze_benchmark_results(suite: BenchmarkSuite):
    """Analyze and display benchmark results"""
    print("\n" + "="*50)
    print("BENCHMARK ANALYSIS")
    print("="*50)
    
    summary = suite.get_summary_stats()
    if summary:
        print(f"Total Results: {summary['total_results']}")
        print(f"Datasets Tested: {summary['datasets_tested']}")
        print(f"Average Latency: {summary['avg_latency']:.3f}s")
        print(f"Average Throughput: {summary['avg_throughput']:.1f} tokens/s")
        print(f"Average Success Rate: {summary['avg_success_rate']:.2%}")
        print(f"Best Throughput: {summary['best_throughput']:.1f} tokens/s")
        print(f"Lowest Latency: {summary['lowest_latency']:.3f}s")
    
    best_result = suite.get_best_performance()
    if best_result:
        print(f"\nBest Performance:")
        print(f"  Dataset: {best_result.dataset_name}")
        print(f"  Batch Size: {best_result.batch_size}")
        print(f"  Throughput: {best_result.avg_throughput:.1f} tokens/s")
        print(f"  Latency: {best_result.avg_latency:.3f}s")
    
    print("="*50)


def validate_results(suite: BenchmarkSuite) -> bool:
    """Validate benchmark results for reasonableness"""
    if not suite.results:
        logger.error("No benchmark results to validate")
        return False
    
    validation_passed = True
    
    for result in suite.results:
        # Basic sanity checks
        if result.avg_latency <= 0:
            logger.error(f"Invalid latency: {result.avg_latency}")
            validation_passed = False
            
        if result.avg_throughput < 0:
            logger.error(f"Invalid throughput: {result.avg_throughput}")
            validation_passed = False
            
        if not 0 <= result.success_rate <= 1:
            logger.error(f"Invalid success rate: {result.success_rate}")
            validation_passed = False
            
        # Reasonable bounds (these are conservative estimates)
        if result.avg_latency > 60:  # More than 1 minute per request seems unreasonable
            logger.warning(f"High latency detected: {result.avg_latency:.3f}s")
            
        if result.avg_throughput > 10000:  # Very high throughput might indicate error
            logger.warning(f"Very high throughput: {result.avg_throughput:.1f} tokens/s")
    
    if validation_passed:
        logger.info("All benchmark results passed validation")
    else:
        logger.error("Some benchmark results failed validation")
    
    return validation_passed


async def main():
    """Main benchmark execution"""
    print("OpenInferencev2 Real-World Benchmark Suite")
    print("Author: Nik Jois (nikjois@llamasearch.ai)")
    print("="*60)
    
    try:
        # Run basic benchmark
        logger.info("Phase 1: Basic comprehensive benchmark")
        basic_suite = await run_basic_benchmark()
        
        # Validate results
        if not validate_results(basic_suite):
            logger.error("Basic benchmark validation failed")
            return 1
        
        # Analyze results
        analyze_benchmark_results(basic_suite)
        
        # Run dataset-specific tests
        logger.info("Phase 2: Dataset-specific benchmarks")
        dataset_results = await run_dataset_specific_benchmark()
        
        if dataset_results:
            print(f"\nDataset-specific results collected for {len(dataset_results)} datasets")
            for dataset_name, results in dataset_results.items():
                print(f"  {dataset_name}: {len(results)} configurations tested")
        
        print("\nBenchmark completed successfully!")
        print("Note: All benchmarks use real open-source datasets when available,")
        print("      falling back to synthetic data for testing when datasets unavailable.")
        
        return 0
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main()) 
"""
OpenInferencev2 Benchmarking Suite
Real performance benchmarks using open-source datasets from Hugging Face
"""

__version__ = "2.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"

from .real_benchmarks import RealWorldBenchmark, DatasetBenchmark
from .performance_metrics import PerformanceMetrics, BenchmarkResults, BenchmarkSuite

__all__ = [
    "RealWorldBenchmark",
    "DatasetBenchmark", 
    "PerformanceMetrics",
    "BenchmarkResults",
    "BenchmarkSuite"
] 
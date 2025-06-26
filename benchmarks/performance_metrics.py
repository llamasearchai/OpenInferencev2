"""
Performance metrics and benchmark results data structures
Author: Nik Jois (nikjois@llamasearch.ai)
"""

import time
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
import json


@dataclass
class PerformanceMetrics:
    """Individual performance measurement"""
    timestamp: float
    latency: float
    tokens_generated: int
    tokens_per_second: float
    memory_usage_mb: float
    gpu_utilization: float
    success: bool
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class BenchmarkResults:
    """Aggregated benchmark results for a specific configuration"""
    dataset_name: str
    batch_size: int
    avg_latency: float
    p95_latency: float
    avg_throughput: float
    total_throughput: float
    success_rate: float
    total_samples: int
    batches_processed: int
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResults':
        """Create from dictionary"""
        return cls(**data)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkResults':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)


class BenchmarkSuite:
    """Collection of benchmark results with analysis capabilities"""
    
    def __init__(self):
        self.results: List[BenchmarkResults] = []
        self.metadata: Dict[str, Any] = {
            "created_at": time.time(),
            "author": "Nik Jois",
            "email": "nikjois@llamasearch.ai",
            "version": "2.0.0"
        }
    
    def add_result(self, result: BenchmarkResults):
        """Add a benchmark result"""
        self.results.append(result)
    
    def get_results_by_dataset(self, dataset_name: str) -> List[BenchmarkResults]:
        """Get all results for a specific dataset"""
        return [r for r in self.results if r.dataset_name == dataset_name]
    
    def get_results_by_batch_size(self, batch_size: int) -> List[BenchmarkResults]:
        """Get all results for a specific batch size"""
        return [r for r in self.results if r.batch_size == batch_size]
    
    def get_best_performance(self) -> Optional[BenchmarkResults]:
        """Get the result with best throughput"""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.avg_throughput)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all results"""
        if not self.results:
            return {}
        
        return {
            "total_results": len(self.results),
            "datasets_tested": len(set(r.dataset_name for r in self.results)),
            "avg_latency": sum(r.avg_latency for r in self.results) / len(self.results),
            "avg_throughput": sum(r.avg_throughput for r in self.results) / len(self.results),
            "avg_success_rate": sum(r.success_rate for r in self.results) / len(self.results),
            "best_throughput": max(r.avg_throughput for r in self.results),
            "lowest_latency": min(r.avg_latency for r in self.results)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire suite to dictionary"""
        return {
            "metadata": self.metadata,
            "results": [r.to_dict() for r in self.results],
            "summary": self.get_summary_stats()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkSuite':
        """Create from dictionary"""
        suite = cls()
        suite.metadata = data.get("metadata", suite.metadata)
        for result_data in data.get("results", []):
            suite.add_result(BenchmarkResults.from_dict(result_data))
        return suite
    
    @classmethod
    def from_json(cls, json_str: str) -> 'BenchmarkSuite':
        """Create from JSON string"""
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, filename: str):
        """Save benchmark suite to file"""
        with open(filename, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'BenchmarkSuite':
        """Load benchmark suite from file"""
        with open(filename, 'r') as f:
            return cls.from_json(f.read()) 
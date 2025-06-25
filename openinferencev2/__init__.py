"""
OpenInferencev2: High-Performance Distributed LLM Inference Engine

A state-of-the-art, production-ready distributed inference engine for Large Language Models
with advanced optimizations, comprehensive monitoring, and enterprise-grade reliability.

Features:
- Custom CUDA kernels with FlashAttention optimization
- Multi-GPU distributed computing with tensor/pipeline parallelism
- Advanced memory management and KV-cache optimization
- Production-ready monitoring and observability
- Enterprise deployment with Docker/Kubernetes support
"""

__version__ = "2.0.0"
__author__ = "Nik Jois"
__email__ = "nikjois@llamasearch.ai"
__license__ = "MIT"
__copyright__ = "Copyright 2024, Nik Jois"
__url__ = "https://github.com/llamasearchai/OpenInferencev2"
__description__ = "High-Performance Distributed LLM Inference Engine"

# Core imports
from .openinferencev2 import (
    OpenInferencev2Engine,
    InferenceRequest,
    InferenceResponse,
    ModelConfig,
)
from .config import Config
from .monitor import PerformanceMonitor
from .scheduler import RequestScheduler
from .optimization import ModelOptimizer

# Version information
VERSION_INFO = {
    "version": __version__,
    "author": __author__,
    "email": __email__,
    "license": __license__,
    "url": __url__,
    "description": __description__,
}

# Export all public APIs
__all__ = [
    # Core classes
    "OpenInferencev2Engine",
    "InferenceRequest", 
    "InferenceResponse",
    "ModelConfig",
    "Config",
    "PerformanceMonitor",
    "RequestScheduler",
    "ModelOptimizer",
    # Version info
    "__version__",
    "VERSION_INFO",
]

# Package metadata for introspection
def get_version():
    """Get the current version of OpenInferencev2."""
    return __version__

def get_info():
    """Get comprehensive package information."""
    return VERSION_INFO.copy()

# Compatibility check
def check_dependencies():
    """Check if all required dependencies are available."""
    import sys
    
    if sys.version_info < (3, 8):
        raise RuntimeError("OpenInferencev2 requires Python 3.8 or later")
    
    try:
        import torch
        if torch.__version__ < "2.0.0":
            print("Warning: PyTorch 2.0+ recommended for optimal performance")
    except ImportError:
        raise RuntimeError("PyTorch is required but not installed")
    
    return True

# Initialize package
check_dependencies()

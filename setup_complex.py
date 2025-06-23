#!/usr/bin/env python3
"""
Setup script for OpenInferencev2: High-Performance Distributed LLM Inference Engine
"""
from setuptools import setup, find_packages, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pathlib import Path
import os

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "docs" / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('--'):
            requirements.append(line)

# Optional C++ extension (only if CUDA is available)
ext_modules = []
cmdclass = {}

try:
    import pybind11
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    import torch
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA detected, building C++ extension...")
        cuda_ext = Pybind11Extension(
            "openinferencev2_cpp",
            [
                "src/core/python_bindings.cpp",
                "src/core/inference_engine.cpp",
                "src/core/distributed_manager.cpp",
            ],
            include_dirs=[
                "src/core",
                "/usr/local/cuda/include",
            ],
            libraries=["cudart", "cublas", "curand", "cusparse"],
            library_dirs=["/usr/local/cuda/lib64"],
            language='c++',
            cxx_std=17,
        )
        ext_modules.append(cuda_ext)
        cmdclass["build_ext"] = build_ext
    else:
        print("CUDA not available, skipping C++ extension.")
        
except ImportError as e:
    print(f"Warning: {e}. Skipping C++ extension.")

setup(
    name="openinferencev2",
    version="2.0.0",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="High-Performance Distributed LLM Inference Engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikjois/openinferencev2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.11.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "torch[cuda]>=2.0.0",
            "nvidia-ml-py>=12.0.0",
        ]
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    entry_points={
        "console_scripts": [
            "openinferencev2=src.cli.main:main",
            "openinferencev2-cli=src.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "openinferencev2": ["*.json", "*.yaml", "*.yml"],
        "docs": ["*.md"],
    },
    zip_safe=False,
    project_urls={
        "Bug Reports": "https://github.com/nikjois/openinferencev2/issues",
        "Source": "https://github.com/nikjois/openinferencev2",
        "Documentation": "https://openinferencev2.readthedocs.io/",
    },
) 
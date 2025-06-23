#!/usr/bin/env python3
"""
Setup script for OpenInferencev2
High-Performance Distributed LLM Inference Engine
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding='utf-8')
except FileNotFoundError:
    long_description = "OpenInferencev2: High-Performance Distributed LLM Inference Engine"

# Core requirements
requirements = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
    "rich>=13.0.0",
    "psutil>=5.9.0",
    "numpy>=1.24.0",
    "accelerate>=0.20.0",
    "tokenizers>=0.13.0",
]

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
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "openinferencev2=src.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="llm, inference, pytorch, cuda, distributed, optimization",
    project_urls={
        "Bug Reports": "https://github.com/nikjois/openinferencev2/issues",
        "Source": "https://github.com/nikjois/openinferencev2",
        "Documentation": "https://github.com/nikjois/openinferencev2/blob/main/README.md",
    },
) 
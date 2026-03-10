# setup.py
"""
Setup script for SentinelML.
"""

import os

from setuptools import find_packages, setup


def read_requirements():
    """Read requirements from requirements.txt."""
    with open("requirements.txt") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def read_readme():
    """Read README.md."""
    if os.path.exists("README.md"):
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    return ""


setup(
    name="sentinelml",
    version="2.0.0",
    author="SentinelML Team",
    author_email="team@sentinelml.ai",
    description="Unified Reliability Engine for AI/ML Systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/sentinelml/sentinelml",
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "joblib>=1.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=5.4.0",
        "requests>=2.26.0",
    ],
    extras_require={
        "torch": ["torch>=1.9.0", "torchvision>=0.10.0"],
        "tensorflow": ["tensorflow>=2.6.0"],
        "transformers": ["transformers>=4.20.0", "tokenizers>=0.12.0"],
        "genai": ["openai>=0.27.0", "anthropic>=0.3.0"],
        "rag": ["langchain>=0.0.200", "llama-index>=0.6.0"],
        "streaming": ["kafka-python>=2.0.0", "confluent-kafka>=1.9.0"],
        "serving": ["fastapi>=0.85.0", "uvicorn>=0.18.0", "grpcio>=1.48.0"],
        "vectorstore": ["faiss-cpu>=1.7.0", "chromadb>=0.3.0", "pinecone-client>=2.0.0"],
        "all": [
            "torch>=1.9.0",
            "tensorflow>=2.6.0",
            "transformers>=4.20.0",
            "openai>=0.27.0",
            "langchain>=0.0.200",
            "llama-index>=0.6.0",
            "fastapi>=0.85.0",
            "uvicorn>=0.18.0",
            "faiss-cpu>=1.7.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=0.950",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sentinelml=sentinelml.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

"""Setup configuration for MCP Standards Server."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="mcp-standards-server",
    version="0.1.0",
    author="William Zujkowski",
    description="A Model Context Protocol server for intelligent access to development standards",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/williamzujkowski/mcp-standards-server",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "pyyaml>=6.0.1",
        "aiohttp>=3.9.0",
        "mcp>=0.1.0",  # Model Context Protocol SDK - critical dependency
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "nltk>=3.8.0",
        "fuzzywuzzy>=0.18.0",
        "python-Levenshtein>=0.12.0",  # For better fuzzy matching performance
        "redis>=4.0.0",
        "tiktoken>=0.5.0",  # For accurate token counting
        # Async support
        "asyncio>=3.4.3",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "pytest-timeout>=2.1.0",
            "pytest-xdist>=3.0.0",  # For parallel test execution
            "memory-profiler>=0.61.0",
            "psutil>=5.9.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "pytest-timeout>=2.1.0",
            "pytest-xdist>=3.0.0",  # For parallel test execution
            "memory-profiler>=0.61.0",
            "psutil>=5.9.0",
            # Add other dev tools if needed
        ],
        "performance": [
            "faiss-cpu>=1.7.0",  # For faster vector search
            "annoy>=1.17.0",  # Alternative vector index
        ],
        "visualization": [
            "matplotlib>=3.5.0",  # For benchmark plots
        ],
        "full": [
            "chromadb>=0.4.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            # Include all optional dependencies
            "faiss-cpu>=1.7.0",
            "annoy>=1.17.0",
            "matplotlib>=3.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mcp-standards=src.cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.md"],
    },
)
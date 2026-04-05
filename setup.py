from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="llm-toolkit",
    version="0.1.0",
    author="simonlpaige",
    description="Python utility library for LLM development — prompt chaining, token counting, cost estimation, retry handling, streaming, and RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simonlpaige/llm-toolkit",
    packages=find_packages(exclude=["examples", "tests"]),
    python_requires=">=3.9",
    install_requires=[
        "openai>=1.30.0",
        "anthropic>=0.25.0",
        "google-generativeai>=0.5.0",
        "tiktoken>=0.7.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.26.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm openai anthropic gemini tokens cost rag prompt chaining",
)

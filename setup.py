from setuptools import setup, find_packages
import os

# Read the README file if it exists
long_description = ""
if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

setup(
    name="deft-enzyme-classification",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="DEFT: Enzyme Classification using Evolutionary Scale Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/deft-enzyme-classification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "datasets>=2.0.0",
        "evaluate>=0.3.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "simple-parsing>=0.1.0",
        "wandb>=0.13.0",
        "biopython>=1.79",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "deft=deft:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json"],
    },
)

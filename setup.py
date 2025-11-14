"""Setup configuration for LLM Evaluation Platform"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = Path("README.md").read_text(encoding="utf-8")

# Read requirements
requirements = Path("requirements.txt").read_text().splitlines()
requirements = [r.strip() for r in requirements if r.strip() and not r.startswith("#")]

setup(
    name="llm-evaluation-platform",
    version="2.0.0",
    author="Rosalina Torres",
    author_email="rosalinatorres888@gmail.com",
    description="Professional platform for evaluating and comparing Large Language Models",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/rosalinatorres888/llm-evaluation-platform",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": Path("requirements-dev.txt").read_text().splitlines()
    },
    entry_points={
        "console_scripts": [
            "llm-eval=scripts.cli:main",
            "llm-dashboard=dashboard.app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

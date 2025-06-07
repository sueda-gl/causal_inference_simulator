"""Setup configuration for the Recommendation Impact Simulator package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="recommendation-impact-simulator",
    version="0.1.0",
    author="AI Visibility Analytics",
    author_email="analytics@aivisibility.com",
    description="Causal inference engine for AI visibility recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aivisibility/recommendation-impact-simulator",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
)
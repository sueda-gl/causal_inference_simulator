"""Recommendation Impact Simulator - Causal inference for AI visibility."""

# Set up logging when package is imported
from .utils import setup_logging

# Initialize logging with default settings
setup_logging(enable_file_logging=False)  # File logging can be enabled by the app

__version__ = "0.1.0"
__author__ = "AI Visibility Analytics"
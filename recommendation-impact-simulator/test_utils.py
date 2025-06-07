#!/usr/bin/env python
"""
Test basic utility functions without causal inference dependencies
"""

import sys
from pathlib import Path

# Add the current directory to path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

try:
    from src.utils.logger import setup_logging, get_logger
    
    # Initialize logging
    setup_logging(enable_file_logging=False)
    logger = get_logger(__name__)
    
    logger.info("Logger imported successfully!")
    print("✅ Logger imported successfully!")
except Exception as e:
    print(f"❌ Logger import failed: {e}")

try:
    from src.utils.helpers import (
        calculate_confidence_interval,
        calculate_summary_statistics,
        format_number,
    )
    
    # Test formatting function
    formatted = format_number(0.12345, 2, '%')
    print(f"✅ Helpers imported successfully! Test: {formatted}")
except Exception as e:
    print(f"❌ Helpers import failed: {e}")

try:
    from src.config.settings import get_settings
    
    # Test settings
    settings = get_settings()
    print(f"✅ Settings imported successfully! App name: {settings.app_name}")
except Exception as e:
    print(f"❌ Settings import failed: {e}") 
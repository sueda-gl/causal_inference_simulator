#!/usr/bin/env python
"""
Test imports to identify the issue.
"""

import sys
from pathlib import Path

# Print the current path for debugging
print("Current sys.path:")
for p in sys.path:
    print(f"  - {p}")

# Add the current directory to path
current_dir = Path(__file__).parent.absolute()
print(f"\nAdding {current_dir} to sys.path")
sys.path.insert(0, str(current_dir))

try:
    print("\nTrying direct import:")
    from src.data.generator import CausalDataGenerator
    print("✅ Direct import successful!")
except Exception as e:
    print(f"❌ Direct import failed: {e}")

try:
    print("\nTrying to import through __init__:")
    from src.data import CausalDataGenerator
    print("✅ Init import successful!")
except Exception as e:
    print(f"❌ Init import failed: {e}")

print("\nTrying to import config:")
try:
    from src.config import get_settings
    print("✅ Config import successful!")
except Exception as e:
    print(f"❌ Config import failed: {e}") 
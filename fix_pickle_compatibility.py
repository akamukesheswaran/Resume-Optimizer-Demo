#!/usr/bin/env python3
"""
Pickle Compatibility Fixer
Fixes numpy._core compatibility issues when loading pickle files
"""

import sys
import pickle
import numpy as np

print("\n" + "="*70)
print("PICKLE COMPATIBILITY FIXER")
print("="*70 + "\n")

print(f"Python version: {sys.version}")
print(f"NumPy version: {np.__version__}")

# Check if we have the compatibility issue
has_core = hasattr(np, '_core')
print(f"Has numpy._core: {has_core}")

if not has_core:
    print("\n⚠️  NumPy version doesn't have _core module")
    print("Installing compatibility patch...\n")
    
    # Create numpy._core as an alias to numpy.core
    import numpy.core as core
    sys.modules['numpy._core'] = core
    sys.modules['numpy._core.multiarray'] = core.multiarray
    sys.modules['numpy._core.umath'] = core.umath
    
    print("✅ Compatibility patch installed!")
else:
    print("\n✅ NumPy version is compatible (has _core module)")

print("\n" + "="*70)
print("Testing pickle loading...")
print("="*70 + "\n")

# Test loading the pickle files
import os

model_files = [
    'models/label_encoder.pkl',
    'models/svm_combined.pkl',
    'models/tfidf_vectorizer_combined.pkl'
]

success_count = 0
for filepath in model_files:
    if not os.path.exists(filepath):
        print(f"❌ {filepath} - NOT FOUND")
        continue
    
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ {filepath} - Loaded successfully!")
        success_count += 1
    except Exception as e:
        print(f"❌ {filepath} - Error: {str(e)[:100]}")

print("\n" + "="*70)
if success_count == 3:
    print("✅ ALL MODELS LOADED SUCCESSFULLY!")
    print("="*70)
    print("\nYou can now run: python app.py")
else:
    print(f"⚠️  Only {success_count}/3 models loaded successfully")
    print("="*70)
    print("\nYou may need to recreate the pickle files with your numpy version.")
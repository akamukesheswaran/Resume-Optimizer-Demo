#!/usr/bin/env python3
"""
Regenerate Pickle Files with Current NumPy Version
This script loads your existing pickle files and re-saves them with your current numpy
"""

import pickle
import sys
import numpy as np

print("\n" + "="*70)
print("PICKLE FILE REGENERATOR")
print("="*70 + "\n")

print(f"Python version: {sys.version.split()[0]}")
print(f"NumPy version: {np.__version__}")
print()

# Files to convert
model_files = [
    'models/label_encoder.pkl',
    'models/svm_combined.pkl',
    'models/tfidf_vectorizer_combined.pkl'
]

print("⚠️  IMPORTANT: This will overwrite your existing pickle files!")
print("Make sure you have backups if needed.\n")

response = input("Continue? (yes/no): ").strip().lower()
if response not in ['yes', 'y']:
    print("Cancelled.")
    sys.exit(0)

print("\nAttempting to load and re-save pickle files...\n")

# Try using pickle protocol 4 which is more compatible
import pickle5 as pickle_loader

success_count = 0
for filepath in model_files:
    try:
        print(f"Processing {filepath}...")
        
        # Try loading with pickle5 for backwards compatibility
        try:
            import pickle5
            with open(filepath, 'rb') as f:
                model = pickle5.load(f)
            print(f"  ✅ Loaded with pickle5")
        except ImportError:
            print("  ⚠️  pickle5 not found, trying standard pickle...")
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print(f"  ✅ Loaded with standard pickle")
        
        # Re-save with current numpy
        backup_path = filepath + '.backup'
        import shutil
        shutil.copy(filepath, backup_path)
        print(f"  💾 Backup saved to {backup_path}")
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f, protocol=4)
        
        print(f"  ✅ Re-saved successfully!\n")
        success_count += 1
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}\n")

print("="*70)
if success_count == 3:
    print("✅ ALL FILES REGENERATED SUCCESSFULLY!")
    print("\nYou can now run: python app.py")
else:
    print(f"⚠️  Only {success_count}/3 files regenerated")
    print("\nSee error messages above for details.")
print("="*70)
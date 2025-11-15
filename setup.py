"""
Resume Matcher - Quick Setup Script
This script helps you set up the Resume Matcher application
"""

import os
import sys
import subprocess

def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70 + "\n")

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_models():
    """Check if model files exist"""
    model_files = [
        'models/label_encoder.pkl',
        'models/svm_combined.pkl',
        'models/tfidf_vectorizer_combined.pkl'
    ]
    
    missing = []
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {file} ({size} bytes)")
        else:
            print(f"❌ {file} - NOT FOUND")
            missing.append(file)
    
    return len(missing) == 0, missing

def check_config():
    """Check if API key is configured"""
    if not os.path.exists('config.py'):
        print("❌ config.py not found")
        return False
    
    with open('config.py', 'r') as f:
        content = f.read()
        if 'your-claude-api-key-here' in content:
            print("⚠️  WARNING: Claude API key not configured in config.py")
            return False
    
    print("✅ config.py exists and API key is set")
    return True

def install_dependencies():
    """Install required packages"""
    print("📦 Installing dependencies from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies")
        return False

def create_uploads_folder():
    """Create uploads folder if it doesn't exist"""
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        print("✅ Created 'uploads' folder")
    else:
        print("✅ 'uploads' folder exists")

def main():
    print_header("RESUME MATCHER - SETUP")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print_header("CHECKING MODEL FILES")
    models_ok, missing = check_models()
    
    if not models_ok:
        print("\n⚠️  MISSING MODEL FILES!")
        print("Please add the following files to the 'models/' folder:")
        for file in missing:
            print(f"   - {file}")
        print("\nYou should have received these files separately.")
        proceed = input("\nDo you want to continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    print_header("CHECKING CONFIGURATION")
    config_ok = check_config()
    
    if not config_ok:
        print("\n⚠️  API KEY NOT CONFIGURED!")
        print("Please edit config.py and add your Claude API key:")
        print("   CLAUDE_API_KEY = 'your-actual-api-key-here'")
        print("\nGet your API key from: https://console.anthropic.com/")
        proceed = input("\nDo you want to continue anyway? (y/n): ")
        if proceed.lower() != 'y':
            sys.exit(1)
    
    print_header("CREATING FOLDERS")
    create_uploads_folder()
    
    print_header("INSTALLING DEPENDENCIES")
    install = input("Install Python dependencies? (y/n): ")
    if install.lower() == 'y':
        if not install_dependencies():
            sys.exit(1)
    
    print_header("SETUP COMPLETE!")
    print("✅ Resume Matcher is ready to use!")
    print("\nTo start the application:")
    print("   python app.py")
    print("\nThen open your browser to:")
    print("   http://127.0.0.1:5000")
    
    start_now = input("\nStart the application now? (y/n): ")
    if start_now.lower() == 'y':
        print("\n🚀 Starting Resume Matcher...")
        os.system('python app.py')

if __name__ == "__main__":
    main()
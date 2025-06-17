import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found")
        print("Please create a .env file with your TOGETHER_API_KEY")
        print("You can copy .env.example to .env and fill in your API key")
        return False
    
    with open(env_path, 'r') as f:
        content = f.read()
        if 'TOGETHER_API_KEY=' in content and 'your_together_api_key_here' not in content:
            print("✅ TOGETHER_API_KEY found in .env file")
            return True
        else:
            print("❌ TOGETHER_API_KEY not properly set in .env file")
            return False

def install_requirements():
    """Install required packages"""
    try:
        print("📦 Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install packages")
        return False

def check_tesseract():
    """Check if Tesseract OCR is available"""
    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, check=True)
        print("✅ Tesseract OCR is available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  Tesseract OCR not found")
        print("Image processing (OCR) will not work without Tesseract")
        print("Download from: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def main():
    """Run setup checks"""
    print("🔧 Data Analyst Agent Setup")
    print("=" * 40)
    
    checks = [
        ("Python Version", check_python_version),
        ("Environment File", check_env_file),
        ("Dependencies", install_requirements),
        ("Tesseract OCR", check_tesseract),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n🔍 Checking {name}...")
        result = check_func()
        results.append((name, result))
    
    print("\n" + "=" * 40)
    print("📋 Setup Summary:")
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Setup complete! You can now run the application:")
        print("  • Command line: python data_analyst_agent.py")
        print("  • Web interface: streamlit run streamlit_app.py")
    else:
        print("\n⚠️  Please fix the failed checks before running the application")

if __name__ == "__main__":
    main()

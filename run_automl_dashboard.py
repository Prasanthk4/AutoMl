#!/usr/bin/env python3
"""
Launch the AutoML Dashboard
Simple script to run the Streamlit AutoML application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit dashboard"""
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    
    # Check if virtual environment exists
    venv_path = script_dir / 'automl_venv'
    if not venv_path.exists():
        print("❌ Virtual environment not found!")
        print("Please run the setup first or activate your virtual environment.")
        sys.exit(1)
    
    # Path to streamlit app
    app_path = script_dir / 'streamlit_app.py'
    if not app_path.exists():
        print("❌ Streamlit app not found!")
        print(f"Expected: {app_path}")
        sys.exit(1)
    
    # Activate virtual environment and run streamlit
    if os.name == 'nt':  # Windows
        activate_cmd = str(venv_path / 'Scripts' / 'activate.bat')
        python_cmd = str(venv_path / 'Scripts' / 'python.exe')
    else:  # Unix/Linux/macOS
        activate_cmd = f"source {venv_path}/bin/activate"
        python_cmd = str(venv_path / 'bin' / 'python')
    
    print("🚀 Starting AutoML Dashboard...")
    print("📱 The dashboard will open in your web browser")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Run streamlit
        if os.name == 'nt':  # Windows
            subprocess.run([python_cmd, '-m', 'streamlit', 'run', str(app_path)], check=True)
        else:  # Unix/Linux/macOS
            subprocess.run([
                'bash', '-c', 
                f"{activate_cmd} && streamlit run {app_path}"
            ], check=True)
            
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thanks for using AutoML System!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
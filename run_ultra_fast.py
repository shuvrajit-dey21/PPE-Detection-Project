#!/usr/bin/env python3
"""
PPE Compliance Monitoring System - Ultra Fast Launcher
Launch the fastest version with cancellation support
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    """Launch ultra-fast version"""
    print("🚀 PPE Monitor Pro - Ultra Fast")
    print("   Lightning speed • Cancellation support • Modern UI")
    print("=" * 55)
    
    # Check files
    if not Path("app_ultra_fast.py").exists():
        print("❌ app_ultra_fast.py not found")
        if Path("app_optimized.py").exists():
            print("📁 Using optimized version instead...")
            app_file = "app_optimized.py"
        elif Path("app.py").exists():
            print("📁 Using standard version instead...")
            app_file = "app.py"
        else:
            print("❌ No application files found")
            return
    else:
        app_file = "app_ultra_fast.py"
    
    # Check model
    if Path("best.pt").exists():
        print("✅ Model file found")
    else:
        print("⚠️  Model file 'best.pt' not found")
        response = input("Continue anyway? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            return
    
    print(f"\n🚀 Starting {app_file}...")
    print("📱 Opening at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    print("-" * 55)
    
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', app_file,
            '--server.port=8501',
            '--browser.gatherUsageStats=false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Stopped")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()

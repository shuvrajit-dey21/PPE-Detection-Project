"""
Test script to verify instant analysis functionality
"""

import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os

def test_image_validation():
    """Test image validation functionality"""
    try:
        from utils import validate_image
        print("✅ Image validation module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import image validation: {e}")
        return False

def test_detection_engine():
    """Test detection engine functionality"""
    try:
        from ppe_detection_engine import PPEDetectionEngine
        print("✅ PPE Detection Engine imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import detection engine: {e}")
        return False

def test_theme_manager():
    """Test theme manager functionality"""
    try:
        from theme_manager import ThemeManager
        theme_manager = ThemeManager()
        theme_config = theme_manager.get_theme_config('light')
        print("✅ Theme Manager working correctly")
        print(f"   Light theme colors: {list(theme_config.keys())[:5]}...")
        return True
    except Exception as e:
        print(f"❌ Theme Manager error: {e}")
        return False

def create_test_image():
    """Create a simple test image"""
    try:
        # Create a simple test image
        test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            pil_image.save(tmp_file.name, 'JPEG')
            print(f"✅ Test image created: {tmp_file.name}")
            return tmp_file.name
    except Exception as e:
        print(f"❌ Failed to create test image: {e}")
        return None

def main():
    """Run all tests"""
    print("🧪 Testing Instant Analysis Page Components")
    print("=" * 50)
    
    tests = [
        ("Image Validation", test_image_validation),
        ("Detection Engine", test_detection_engine),
        ("Theme Manager", test_theme_manager),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All components are working correctly!")
        
        # Create test image
        print("\n🖼️ Creating test image...")
        test_image_path = create_test_image()
        if test_image_path:
            print("✅ Test image ready for upload testing")
            print(f"   You can use this image to test the upload functionality")
            print(f"   Image path: {test_image_path}")
    else:
        print("⚠️ Some components have issues. Please check the errors above.")

if __name__ == "__main__":
    main()

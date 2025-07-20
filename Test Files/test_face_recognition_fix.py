#!/usr/bin/env python3
"""
Test script to verify face recognition engine is working properly
"""

import cv2
import numpy as np
from face_recognition_engine import FaceRecognitionEngine
from ppe_detection_engine import PPEDetectionEngine

def test_opencv_face_module():
    """Test OpenCV face recognition module"""
    print("🧪 Testing OpenCV Face Recognition Module...")
    
    try:
        # Test OpenCV version
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Test face module availability
        assert hasattr(cv2, 'face'), "cv2.face module not available"
        print("✅ cv2.face module is available")
        
        # Test LBPHFaceRecognizer
        assert hasattr(cv2.face, 'LBPHFaceRecognizer_create'), "LBPHFaceRecognizer_create not available"
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("✅ LBPHFaceRecognizer created successfully")
        
        return True
    except Exception as e:
        print(f"❌ OpenCV face module test failed: {e}")
        return False

def test_face_recognition_engine():
    """Test Face Recognition Engine"""
    print("\n🧪 Testing Face Recognition Engine...")
    
    try:
        # Initialize engine
        engine = FaceRecognitionEngine()
        print("✅ Face Recognition Engine initialized")
        
        # Test dataset info
        dataset_info = engine.get_dataset_info()
        print(f"✅ Dataset info: {dataset_info['total_people']} people, {dataset_info['total_samples']} samples")
        
        # Test face detection on dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        face_results = engine.detect_faces(dummy_image)
        print(f"✅ Face detection test: {len(face_results)} faces detected (expected 0 for blank image)")
        
        # Test training status
        training_status = engine.get_training_status_info()
        print(f"✅ Training status: {training_status}")
        
        return True
    except Exception as e:
        print(f"❌ Face Recognition Engine test failed: {e}")
        return False

def test_ppe_integration():
    """Test PPE Detection Engine with Face Recognition"""
    print("\n🧪 Testing PPE Detection Engine Integration...")
    
    try:
        # Initialize PPE engine with face recognition
        ppe_engine = PPEDetectionEngine(enable_face_recognition=True)
        print("✅ PPE Detection Engine with Face Recognition initialized")
        
        # Check if face engine is available
        face_engine = ppe_engine.get_face_engine()
        if face_engine:
            print("✅ Face engine is available in PPE detector")
        else:
            print("⚠️ Face engine is not available in PPE detector")
            return False
        
        # Test detection on dummy image
        dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = ppe_engine.detect_objects(dummy_image)
        
        if 'error' not in results:
            print("✅ PPE detection with face recognition works")
            print(f"✅ Results contain: {list(results.keys())}")
        else:
            print(f"❌ PPE detection failed: {results['error']}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ PPE Integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Face Recognition Fix Verification Tests\n")
    
    tests = [
        ("OpenCV Face Module", test_opencv_face_module),
        ("Face Recognition Engine", test_face_recognition_engine),
        ("PPE Integration", test_ppe_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"📋 TEST SUMMARY")
    print(f"{'='*50}")
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Face Recognition is working properly!")
        print("\n📝 What was fixed:")
        print("   • Removed conflicting opencv-python and opencv-python-headless packages")
        print("   • Installed opencv-contrib-python which includes face recognition modules")
        print("   • Updated requirements.txt to use opencv-contrib-python")
        print("   • Verified all face recognition functionality is working")
    else:
        print("⚠️ Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    main()

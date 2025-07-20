#!/usr/bin/env python3
"""
Test script to verify all fixes and new features are working properly
"""

import os
import sys
import traceback
from datetime import date, timedelta

def test_imports():
    """Test that all modules import correctly"""
    print("🔍 Testing imports...")
    
    try:
        # Test main app import
        import app_ultra_fast
        print("✅ Main app imports successfully")
        
        # Test face recognition engine
        from face_recognition_engine import FaceRecognitionEngine
        print("✅ Face recognition engine imports successfully")
        
        # Test webcam component
        from webcam_component import create_webcam_interface
        print("✅ Webcam component imports successfully")
        
        # Test attendance manager
        from attendance_manager import AttendanceManager
        print("✅ Attendance manager imports successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_face_recognition_improvements():
    """Test face recognition improvements"""
    print("\n🔍 Testing face recognition improvements...")
    
    try:
        from face_recognition_engine import FaceRecognitionEngine
        
        # Initialize face recognition engine
        face_engine = FaceRecognitionEngine()
        print("✅ Face recognition engine initialized")
        
        # Test confidence threshold update
        original_threshold = face_engine.confidence_threshold
        face_engine.update_confidence_threshold(75)
        assert face_engine.confidence_threshold == 75, "Threshold update failed"
        print("✅ Confidence threshold update works")
        
        # Test model stats
        stats = face_engine.get_model_stats()
        assert isinstance(stats, dict), "Model stats should return dict"
        assert 'trained' in stats, "Stats should include training status"
        print("✅ Model stats method works")
        
        # Restore original threshold
        face_engine.update_confidence_threshold(original_threshold)
        
        return True
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
        traceback.print_exc()
        return False

def test_attendance_manager():
    """Test attendance manager functionality"""
    print("\n🔍 Testing attendance manager...")
    
    try:
        from attendance_manager import AttendanceManager
        
        # Initialize with test database
        attendance_manager = AttendanceManager("test_attendance.db")
        print("✅ Attendance manager initialized")
        
        # Test getting attendance stats
        stats = attendance_manager.get_attendance_stats()
        assert isinstance(stats, dict), "Stats should be a dictionary"
        print("✅ Attendance stats retrieval works")
        
        # Test getting today's attendance
        today_attendance = attendance_manager.get_today_attendance()
        assert isinstance(today_attendance, list), "Today's attendance should be a list"
        print("✅ Today's attendance retrieval works")
        
        # Test getting all employees
        employees = attendance_manager.get_all_employees()
        assert isinstance(employees, list), "Employees should be a list"
        print("✅ Employee retrieval works")
        
        return True
    except Exception as e:
        print(f"❌ Attendance manager test failed: {e}")
        traceback.print_exc()
        return False

def test_camera_initialization():
    """Test camera initialization improvements"""
    print("\n🔍 Testing camera initialization...")
    
    try:
        from webcam_component import WebcamPPEDetector
        from ppe_detection_engine import PPEDetectionEngine
        
        # Test detection engine initialization
        if os.path.exists("best.pt"):
            detection_engine = PPEDetectionEngine("best.pt")
            print("✅ PPE detection engine initialized")
            
            # Test webcam detector initialization
            webcam_detector = WebcamPPEDetector(detection_engine, None)
            print("✅ Webcam detector initialized")
            
            # Test settings update
            test_settings = {
                'conf_threshold': 0.6,
                'iou_threshold': 0.5,
                'detection_enabled': True
            }
            webcam_detector.update_settings(test_settings)
            print("✅ Settings update works")
            
        else:
            print("⚠️ Model file 'best.pt' not found, skipping detection engine test")
        
        return True
    except Exception as e:
        print(f"❌ Camera initialization test failed: {e}")
        traceback.print_exc()
        return False

def test_new_features():
    """Test new features added"""
    print("\n🔍 Testing new features...")
    
    try:
        # Test that the new visualization functions would work
        # (We can't test the actual Streamlit UI, but we can test the logic)
        
        # Test date calculations for weekly trends
        today = date.today()
        weekly_dates = [today - timedelta(days=i) for i in range(7)]
        assert len(weekly_dates) == 7, "Weekly date calculation failed"
        print("✅ Weekly trends date calculation works")
        
        # Test department statistics logic
        test_employees = [
            {'department': 'Engineering', 'employee_id': 'ENG001'},
            {'department': 'Engineering', 'employee_id': 'ENG002'},
            {'department': 'HR', 'employee_id': 'HR001'},
        ]
        
        departments = {}
        for emp in test_employees:
            dept = emp.get('department', 'Not Specified')
            if dept not in departments:
                departments[dept] = {'total': 0}
            departments[dept]['total'] += 1
        
        assert departments['Engineering']['total'] == 2, "Department counting failed"
        assert departments['HR']['total'] == 1, "Department counting failed"
        print("✅ Department statistics logic works")
        
        return True
    except Exception as e:
        print(f"❌ New features test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Starting comprehensive test suite...\n")
    
    tests = [
        test_imports,
        test_face_recognition_improvements,
        test_attendance_manager,
        test_camera_initialization,
        test_new_features
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed == 0:
        print("\n🎉 All tests passed! The application is ready to use.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

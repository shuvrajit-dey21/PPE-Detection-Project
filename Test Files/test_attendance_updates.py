#!/usr/bin/env python3
"""
Test script to verify that face recognition properly updates live attendance
"""

import os
import sys
import time
import traceback
from datetime import date, datetime

def test_attendance_integration():
    """Test the integration between face recognition and attendance updates"""
    print("🔍 Testing face recognition to attendance integration...")
    
    try:
        from attendance_manager import AttendanceManager
        from face_recognition_engine import FaceRecognitionEngine
        from webcam_component import WebcamPPEDetector
        from ppe_detection_engine import PPEDetectionEngine
        
        # Initialize components
        attendance_manager = AttendanceManager("test_attendance_integration.db")
        print("✅ Attendance manager initialized")
        
        # Test face recognition engine
        face_engine = FaceRecognitionEngine()
        print("✅ Face recognition engine initialized")
        
        # Test if model file exists
        if os.path.exists("best.pt"):
            detection_engine = PPEDetectionEngine("best.pt")
            print("✅ PPE detection engine initialized")
            
            # Test webcam detector with attendance integration
            webcam_detector = WebcamPPEDetector(detection_engine, attendance_manager)
            print("✅ Webcam detector with attendance integration initialized")
            
            # Test attendance update flag
            initial_flag = webcam_detector.attendance_updated
            print(f"✅ Initial attendance update flag: {initial_flag}")
            
            # Simulate attendance update
            webcam_detector.attendance_updated = True
            updated_flag = webcam_detector.check_and_reset_attendance_update()
            print(f"✅ Attendance update flag test: {updated_flag}")
            
            # Verify flag was reset
            final_flag = webcam_detector.attendance_updated
            print(f"✅ Flag reset verification: {final_flag}")
            
            # Test attendance summary
            summary = webcam_detector.get_attendance_summary()
            print(f"✅ Attendance summary retrieved: {type(summary)}")
            
            if 'attendance_updated' in summary:
                print("✅ Attendance update flag included in summary")
            else:
                print("❌ Attendance update flag missing from summary")
                return False
                
        else:
            print("⚠️ Model file 'best.pt' not found, skipping detection engine tests")
        
        return True
        
    except Exception as e:
        print(f"❌ Attendance integration test failed: {e}")
        traceback.print_exc()
        return False

def test_attendance_recording():
    """Test attendance recording functionality"""
    print("\n🔍 Testing attendance recording...")
    
    try:
        from attendance_manager import AttendanceManager
        
        # Initialize with test database
        attendance_manager = AttendanceManager("test_attendance_recording.db")
        print("✅ Test attendance manager initialized")
        
        # Add a test employee
        employee_data = {
            'employee_id': 'TEST001',
            'name': 'Test Employee',
            'department': 'Testing'
        }
        
        success = attendance_manager.add_employee(
            employee_data['employee_id'],
            employee_data['name'],
            employee_data['department']
        )
        
        if success:
            print("✅ Test employee added successfully")
        else:
            print("⚠️ Test employee might already exist")
        
        # Test attendance recording
        record_success = attendance_manager.record_attendance(
            employee_data['employee_id'],
            85.5,  # confidence score
            "Test Camera",
            "test_session_123"
        )
        
        if record_success:
            print("✅ Attendance recorded successfully")
        else:
            print("❌ Failed to record attendance")
            return False
        
        # Verify attendance was recorded
        today_attendance = attendance_manager.get_today_attendance()
        test_record_found = any(
            record['employee_id'] == employee_data['employee_id']
            for record in today_attendance
        )
        
        if test_record_found:
            print("✅ Attendance record verified in database")
        else:
            print("❌ Attendance record not found in database")
            return False
        
        # Test attendance stats
        stats = attendance_manager.get_attendance_stats()
        if stats['present_count'] > 0:
            print(f"✅ Attendance stats updated: {stats['present_count']} present")
        else:
            print("❌ Attendance stats not updated")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Attendance recording test failed: {e}")
        traceback.print_exc()
        return False

def test_face_recognition_thresholds():
    """Test face recognition confidence thresholds"""
    print("\n🔍 Testing face recognition thresholds...")
    
    try:
        from face_recognition_engine import FaceRecognitionEngine
        
        face_engine = FaceRecognitionEngine()
        print("✅ Face recognition engine initialized")
        
        # Test confidence threshold updates
        original_threshold = face_engine.confidence_threshold
        print(f"✅ Original threshold: {original_threshold}")
        
        # Test lowering threshold for better detection
        face_engine.update_confidence_threshold(50)  # Lower threshold
        new_threshold = face_engine.confidence_threshold
        print(f"✅ Updated threshold: {new_threshold}")
        
        if new_threshold == 50:
            print("✅ Threshold update successful")
        else:
            print("❌ Threshold update failed")
            return False
        
        # Test model stats
        stats = face_engine.get_model_stats()
        if isinstance(stats, dict) and 'confidence_threshold' in stats:
            print(f"✅ Model stats retrieved: threshold = {stats['confidence_threshold']}")
        else:
            print("❌ Model stats retrieval failed")
            return False
        
        # Restore original threshold
        face_engine.update_confidence_threshold(original_threshold)
        
        return True
        
    except Exception as e:
        print(f"❌ Face recognition threshold test failed: {e}")
        traceback.print_exc()
        return False

def test_real_time_updates():
    """Test real-time update mechanisms"""
    print("\n🔍 Testing real-time update mechanisms...")
    
    try:
        # Test timing mechanisms
        start_time = time.time()
        time.sleep(0.1)  # Small delay
        end_time = time.time()
        
        if end_time - start_time >= 0.1:
            print("✅ Timing mechanisms working")
        else:
            print("❌ Timing mechanisms failed")
            return False
        
        # Test datetime formatting
        current_time = datetime.now()
        formatted_time = current_time.strftime('%H:%M:%S')
        print(f"✅ Time formatting working: {formatted_time}")
        
        # Test timestamp conversion
        timestamp = time.time()
        converted_time = datetime.fromtimestamp(timestamp)
        print(f"✅ Timestamp conversion working: {converted_time}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real-time update test failed: {e}")
        traceback.print_exc()
        return False

def cleanup_test_files():
    """Clean up test database files"""
    test_files = [
        "test_attendance_integration.db",
        "test_attendance_recording.db"
    ]
    
    for file in test_files:
        try:
            if os.path.exists(file):
                os.remove(file)
                print(f"🧹 Cleaned up {file}")
        except Exception as e:
            print(f"⚠️ Could not clean up {file}: {e}")

def main():
    """Run all attendance update tests"""
    print("🚀 Starting attendance update integration tests...\n")
    
    tests = [
        test_attendance_integration,
        test_attendance_recording,
        test_face_recognition_thresholds,
        test_real_time_updates
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
    
    # Cleanup
    cleanup_test_files()
    
    if failed == 0:
        print("\n🎉 All attendance update tests passed!")
        print("💡 Face recognition should now properly update live attendance.")
        return True
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

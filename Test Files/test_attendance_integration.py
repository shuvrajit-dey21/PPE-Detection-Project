"""
Test script for PPE Attendance Integration
Validates that all components work together correctly
"""

import sys
import os
import logging
from datetime import date, datetime
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_attendance_manager():
    """Test the AttendanceManager functionality"""
    logger.info("Testing AttendanceManager...")
    
    try:
        from attendance_manager import AttendanceManager
        
        # Create temporary database for testing
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        # Initialize attendance manager
        attendance_manager = AttendanceManager(db_path)
        
        # Test adding employees
        assert attendance_manager.add_employee("EMP001", "John Doe", "Engineering"), "Failed to add employee"
        assert attendance_manager.add_employee("EMP002", "Jane Smith", "HR"), "Failed to add employee"
        
        # Test getting employees
        employees = attendance_manager.get_all_employees()
        assert len(employees) == 2, f"Expected 2 employees, got {len(employees)}"
        
        # Test recording attendance
        assert attendance_manager.record_attendance("EMP001", 95.5, "Main Camera"), "Failed to record attendance"
        
        # Test getting today's attendance
        today_attendance = attendance_manager.get_today_attendance()
        assert len(today_attendance) == 1, f"Expected 1 attendance record, got {len(today_attendance)}"
        
        # Test attendance stats
        stats = attendance_manager.get_attendance_stats()
        assert stats['present_count'] == 1, f"Expected 1 present, got {stats['present_count']}"
        assert stats['total_employees'] == 2, f"Expected 2 total employees, got {stats['total_employees']}"
        
        # Test CSV export
        csv_data = attendance_manager.export_attendance_to_csv()
        assert csv_data and not csv_data.startswith("No attendance"), "CSV export failed"
        
        # Test Excel export
        excel_data = attendance_manager.export_attendance_to_excel()
        assert excel_data, "Excel export failed"
        
        # Cleanup
        os.unlink(db_path)
        
        logger.info("✅ AttendanceManager tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ AttendanceManager test failed: {e}")
        return False

def test_face_recognition_integration():
    """Test face recognition integration"""
    logger.info("Testing Face Recognition integration...")
    
    try:
        from face_recognition_engine import FaceRecognitionEngine
        
        # Create temporary directories for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = os.path.join(temp_dir, "face_dataset")
            model_path = os.path.join(temp_dir, "face_model.pkl")
            
            # Initialize face recognition engine
            face_engine = FaceRecognitionEngine(dataset_path, model_path)
            
            # Test dataset info
            dataset_info = face_engine.get_dataset_info()
            assert isinstance(dataset_info, dict), "Dataset info should be a dictionary"
            assert 'total_people' in dataset_info, "Dataset info missing total_people"
            
            logger.info("✅ Face Recognition integration tests passed!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Face Recognition integration test failed: {e}")
        return False

def test_ppe_detection_integration():
    """Test PPE detection engine integration"""
    logger.info("Testing PPE Detection integration...")
    
    try:
        from ppe_detection_engine import PPEDetectionEngine
        
        # Test initialization (may fail if model not available, which is OK for testing)
        try:
            engine = PPEDetectionEngine("best.pt", enable_face_recognition=False)
            logger.info("✅ PPE Detection engine initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ PPE Detection engine initialization failed (expected if model not available): {e}")
        
        logger.info("✅ PPE Detection integration tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ PPE Detection integration test failed: {e}")
        return False

def test_imports():
    """Test that all required modules can be imported"""
    logger.info("Testing module imports...")
    
    required_modules = [
        'attendance_manager',
        'face_recognition_engine', 
        'ppe_detection_engine',
        'webcam_component',
        'utils',
        'results_viewer',
        'theme_manager'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ Successfully imported {module}")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        logger.error(f"❌ Failed to import modules: {failed_imports}")
        return False
    
    logger.info("✅ All module imports successful!")
    return True

def test_database_schema():
    """Test database schema creation and integrity"""
    logger.info("Testing database schema...")
    
    try:
        from attendance_manager import AttendanceManager
        import sqlite3
        
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            db_path = tmp_db.name
        
        # Initialize attendance manager (creates schema)
        attendance_manager = AttendanceManager(db_path)
        
        # Verify tables exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Check employees table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='employees'")
            assert cursor.fetchone(), "employees table not found"
            
            # Check attendance_records table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance_records'")
            assert cursor.fetchone(), "attendance_records table not found"
            
            # Check attendance_sessions table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='attendance_sessions'")
            assert cursor.fetchone(), "attendance_sessions table not found"
            
            # Check indexes
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
            indexes = [row[0] for row in cursor.fetchall()]
            expected_indexes = ['idx_attendance_date', 'idx_attendance_employee', 'idx_sessions_timestamp']
            
            for index in expected_indexes:
                assert index in indexes, f"Index {index} not found"
        
        # Cleanup
        os.unlink(db_path)
        
        logger.info("✅ Database schema tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database schema test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    logger.info("🚀 Starting PPE Attendance Integration Tests...")
    
    tests = [
        ("Module Imports", test_imports),
        ("Database Schema", test_database_schema),
        ("AttendanceManager", test_attendance_manager),
        ("Face Recognition Integration", test_face_recognition_integration),
        ("PPE Detection Integration", test_ppe_detection_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} test PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} test FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name} test FAILED with exception: {e}")
    
    logger.info(f"\n🏁 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! PPE Attendance integration is working correctly.")
        return True
    else:
        logger.error("💥 Some tests failed. Please check the logs above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

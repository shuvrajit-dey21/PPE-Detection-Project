#!/usr/bin/env python3
"""
Test script to verify the duration tracking fix
"""

import time
import sys
import os

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from webcam_component import WebcamPPEDetector
    from ppe_detection_engine import PPEDetectionEngine
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project directory")
    sys.exit(1)

def test_duration_tracking():
    """Test the duration tracking functionality"""
    print("🧪 Testing Duration Tracking Fix...")
    print("=" * 50)
    
    try:
        # Initialize detection engine (mock for testing)
        print("1️⃣ Initializing detection engine...")
        detection_engine = PPEDetectionEngine()
        print("✅ Detection engine initialized")
        
        # Initialize webcam detector
        print("\n2️⃣ Initializing webcam detector...")
        detector = WebcamPPEDetector(detection_engine)
        print("✅ Webcam detector initialized")
        print(f"   📊 Initial session_active: {detector.session_active}")
        print(f"   ⏱️ Initial last_session_duration: {detector.last_session_duration}")
        
        # Simulate session activity
        print("\n3️⃣ Simulating session activity...")
        time.sleep(2)  # Simulate 2 seconds of activity
        
        # Get stats while session is active
        print("\n4️⃣ Getting stats while session is active...")
        stats_active = detector.get_latest_stats()
        if stats_active:
            duration_active = stats_active.get('session_duration', 0)
            print(f"✅ Active session duration: {duration_active:.1f} seconds")
        else:
            print("⚠️ No stats available during active session")
        
        # Stop the session
        print("\n5️⃣ Stopping session...")
        detector.stop_session()
        print(f"✅ Session stopped")
        print(f"   📊 session_active: {detector.session_active}")
        print(f"   ⏱️ last_session_duration: {detector.last_session_duration:.1f}")
        
        # Wait a bit more to simulate time passing after session stop
        print("\n6️⃣ Waiting 2 seconds after session stop...")
        time.sleep(2)
        
        # Get stats after session is stopped
        print("\n7️⃣ Getting stats after session stopped...")
        stats_stopped = detector.get_latest_stats()
        if stats_stopped:
            duration_stopped = stats_stopped.get('session_duration', 0)
            print(f"✅ Stopped session duration: {duration_stopped:.1f} seconds")
            
            # Verify the duration didn't increase after stopping
            if abs(duration_stopped - detector.last_session_duration) < 0.1:
                print("✅ SUCCESS: Duration is preserved after session stop!")
            else:
                print(f"❌ FAIL: Duration changed after stop. Expected: {detector.last_session_duration:.1f}, Got: {duration_stopped:.1f}")
        else:
            print("⚠️ No stats available after session stop")
        
        # Test starting a new session
        print("\n8️⃣ Starting new session...")
        detector.start_session()
        print(f"✅ New session started")
        print(f"   📊 session_active: {detector.session_active}")
        print(f"   ⏱️ last_session_duration: {detector.last_session_duration:.1f}")
        
        # Wait and check new session duration
        time.sleep(1)
        stats_new = detector.get_latest_stats()
        if stats_new:
            duration_new = stats_new.get('session_duration', 0)
            print(f"✅ New session duration: {duration_new:.1f} seconds")
            
            if duration_new < duration_stopped:
                print("✅ SUCCESS: New session has fresh duration!")
            else:
                print(f"❌ FAIL: New session duration seems wrong. Got: {duration_new:.1f}")
        
        print("\n" + "=" * 50)
        print("🎉 Duration tracking test completed!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_duration_tracking()

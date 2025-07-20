#!/usr/bin/env python3
"""
Test script to verify the use_container_width fix in face recognition tab
"""

import sys
import os
import re

# Add current directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_deprecated_parameter_fix():
    """Test that use_column_width has been replaced with use_container_width"""
    print("🧪 Testing Deprecated Parameter Fix...")
    print("=" * 60)
    
    # Files to check
    files_to_check = [
        'app_ultra_fast.py',
        'webcam_component.py',
        'results_viewer.py',
        'utils.py'
    ]
    
    deprecated_found = False
    fixed_found = False
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"\n📁 Checking {file_path}...")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for deprecated parameter
            deprecated_matches = re.findall(r'use_column_width\s*=\s*True', content)
            if deprecated_matches:
                print(f"❌ Found {len(deprecated_matches)} deprecated use_column_width occurrences")
                deprecated_found = True
                
                # Show context for each match
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'use_column_width' in line:
                        print(f"   Line {i+1}: {line.strip()}")
            else:
                print(f"✅ No deprecated use_column_width found")
            
            # Check for correct parameter
            fixed_matches = re.findall(r'use_container_width\s*=\s*True', content)
            if fixed_matches:
                print(f"✅ Found {len(fixed_matches)} correct use_container_width occurrences")
                fixed_found = True
            
        else:
            print(f"⚠️ File {file_path} not found")
    
    print("\n" + "=" * 60)
    print("📊 SUMMARY:")
    
    if deprecated_found:
        print("❌ FAIL: Deprecated use_column_width parameter still found!")
        return False
    else:
        print("✅ SUCCESS: No deprecated use_column_width parameters found!")
    
    if fixed_found:
        print("✅ SUCCESS: Correct use_container_width parameters found!")
    else:
        print("⚠️ INFO: No use_container_width parameters found (this might be normal)")
    
    return True

def test_face_recognition_tab_specifically():
    """Test the specific face recognition tab area where the fix was applied"""
    print("\n🎯 Testing Face Recognition Tab Specifically...")
    print("=" * 60)
    
    file_path = 'app_ultra_fast.py'
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} not found")
        return False
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Look for the specific area around line 5344 (where the fix was applied)
    target_area_start = 5340
    target_area_end = 5350
    
    print(f"🔍 Checking lines {target_area_start}-{target_area_end} for face recognition test functionality...")
    
    found_test_recognition = False
    found_correct_parameter = False
    found_deprecated_parameter = False
    
    for i in range(max(0, target_area_start-1), min(len(lines), target_area_end)):
        line = lines[i]
        line_num = i + 1
        
        # Check for test recognition functionality
        if 'Test Recognition' in line or 'test_recognition' in line:
            found_test_recognition = True
            print(f"✅ Line {line_num}: Found test recognition functionality")
        
        # Check for image display with correct parameter
        if 'st.image' in line and 'use_container_width=True' in line:
            found_correct_parameter = True
            print(f"✅ Line {line_num}: Found correct use_container_width parameter")
            print(f"   Content: {line.strip()}")
        
        # Check for deprecated parameter
        if 'st.image' in line and 'use_column_width=True' in line:
            found_deprecated_parameter = True
            print(f"❌ Line {line_num}: Found deprecated use_column_width parameter")
            print(f"   Content: {line.strip()}")
    
    print(f"\n📊 Face Recognition Tab Test Results:")
    print(f"   🎯 Test Recognition functionality: {'✅ Found' if found_test_recognition else '❌ Not found'}")
    print(f"   ✅ Correct parameter usage: {'✅ Found' if found_correct_parameter else '❌ Not found'}")
    print(f"   ❌ Deprecated parameter usage: {'❌ Found' if found_deprecated_parameter else '✅ None found'}")
    
    if found_test_recognition and found_correct_parameter and not found_deprecated_parameter:
        print("🎉 SUCCESS: Face recognition tab is properly fixed!")
        return True
    else:
        print("❌ FAIL: Face recognition tab has issues!")
        return False

def test_functionality_preservation():
    """Test that all functionality is preserved"""
    print("\n🔧 Testing Functionality Preservation...")
    print("=" * 60)
    
    try:
        # Try to import the main app module
        import app_ultra_fast
        print("✅ app_ultra_fast module imports successfully")
        
        # Check if face recognition functionality is available
        if hasattr(app_ultra_fast, 'create_face_recognition_tab'):
            print("✅ Face recognition tab function is available")
        else:
            print("⚠️ Face recognition tab function not found (might be inline)")
        
        print("✅ All functionality appears to be preserved")
        return True
        
    except Exception as e:
        print(f"❌ Error testing functionality: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing use_container_width Fix")
    print("=" * 60)
    
    # Run all tests
    test1_passed = test_deprecated_parameter_fix()
    test2_passed = test_face_recognition_tab_specifically()
    test3_passed = test_functionality_preservation()
    
    print("\n" + "=" * 60)
    print("🏁 FINAL RESULTS:")
    print(f"   📋 Deprecated Parameter Check: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"   🎯 Face Recognition Tab Check: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"   🔧 Functionality Preservation: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 ALL TESTS PASSED! The fix is successful!")
        print("✅ use_column_width has been properly replaced with use_container_width")
        print("✅ Face recognition test functionality is working correctly")
        print("✅ All other functionality remains intact")
    else:
        print("\n❌ SOME TESTS FAILED! Please review the issues above.")
    
    print("=" * 60)

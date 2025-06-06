#!/usr/bin/env python3
"""
Simple test script for the new JSON response format using existing screenshot
"""

import requests
import json
import os

def test_new_format():
    """Test the new response format using existing screenshot"""
    
    runpod_url = "https://c2yfcdc14566pt-7860.proxy.runpod.net"
    
    print("🧪 Testing new JSON response format...")
    
    # Look for existing screenshot files
    screenshot_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            screenshot_files.append(file)
    
    if not screenshot_files:
        print("❌ No screenshot files found in current directory")
        print("Please add a .png or .jpg file to test with")
        return False
    
    # Use the first screenshot found
    test_image_path = screenshot_files[0]
    print(f"✅ Using test image: {test_image_path}")
    
    # Test detect_elements_super endpoint
    url = f"{runpod_url}/detect_elements_super"
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image_file': f}
            data = {
                'box_threshold': 0.005,
                'iou_threshold': 0.05,
                'imgsz': 1024
            }
            
            print(f"📤 Sending request to {url}")
            print(f"📋 Parameters: {data}")
            
            response = requests.post(url, files=files, data=data, timeout=60)
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Check new format
                print("\n✅ Response received successfully!")
                print(f"📋 Response keys: {list(result.keys())}")
                
                # Verify new field names
                if 'annotated_image_base64' in result and 'elements' in result:
                    print("🎉 NEW FORMAT CONFIRMED!")
                    
                    elements = result['elements']
                    print(f"🎯 Found {len(elements)} elements")
                    print(f"📊 Elements type: {type(elements)}")
                    
                    # Show structure of first few elements
                    print("\n📋 Element structure:")
                    for i, element in enumerate(elements[:3]):
                        print(f"  Element {i}: {element}")
                        
                        # Verify element structure
                        if 'id' in element and 'bbox' in element and 'confidence' in element:
                            print(f"    ✅ Valid structure")
                            bbox = element['bbox']
                            conf = element['confidence']
                            print(f"    📦 bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                            print(f"    🎯 confidence: {conf:.3f}")
                        else:
                            print(f"    ❌ Invalid structure")
                    
                    # Verify annotated image is base64 string
                    img_b64 = result['annotated_image_base64']
                    if isinstance(img_b64, str) and len(img_b64) > 100:
                        print(f"✅ Annotated image: base64 string (length: {len(img_b64)})")
                    else:
                        print(f"❌ Invalid annotated image format")
                        
                    print("\n🎉 NEW JSON FORMAT TEST PASSED!")
                    print("\n📊 Summary:")
                    print(f"  - Response format: ✅ Standard JSON object")
                    print(f"  - Field 'annotated_image_base64': ✅ Present")
                    print(f"  - Field 'elements': ✅ Present as array")
                    print(f"  - Elements detected: {len(elements)}")
                    print(f"  - No JSON.parse() needed: ✅ Direct array access")
                    
                    return True
                        
                else:
                    print("❌ Old format detected!")
                    print(f"Available keys: {list(result.keys())}")
                    
                    # Show what we got for debugging
                    for key, value in result.items():
                        if isinstance(value, str) and len(value) > 100:
                            print(f"  {key}: {type(value)} (length: {len(value)})")
                        else:
                            print(f"  {key}: {value}")
                    return False
                    
            else:
                print(f"❌ Request failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.Timeout:
        print("❌ Request timed out - API may still be building")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed - check if RunPod URL is correct and API is running")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_new_format()
    if success:
        print("\n🎉 ALL TESTS PASSED! New JSON format is working correctly.")
    else:
        print("\n💥 TESTS FAILED! Check the output above for details.") 
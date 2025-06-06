#!/usr/bin/env python3
"""
Simple test script for the new JSON response format
"""

import requests
import json
import base64
from PIL import Image, ImageDraw
import io
import os

def create_test_image():
    """Create a simple test image with some rectangles"""
    # Create a simple test image
    img = Image.new('RGB', (400, 300), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some rectangles to simulate UI elements
    draw.rectangle([50, 50, 150, 100], fill='blue', outline='black', width=2)
    draw.rectangle([200, 80, 350, 130], fill='red', outline='black', width=2)
    draw.rectangle([100, 180, 250, 220], fill='green', outline='black', width=2)
    
    # Add some text
    try:
        draw.text((60, 65), "Button 1", fill='white')
        draw.text((210, 95), "Button 2", fill='white')
        draw.text((110, 190), "Button 3", fill='white')
    except:
        pass  # If font not available, skip text
    
    return img

def test_new_format():
    """Test the new response format"""
    
    runpod_url = "https://c2yfcdc14566pt-7860.proxy.runpod.net"
    
    print("🧪 Testing new JSON response format...")
    
    # Create test image
    test_image = create_test_image()
    
    # Save test image
    os.makedirs('test_images', exist_ok=True)
    test_image_path = "test_images/simple_test.png"
    test_image.save(test_image_path)
    print(f"✅ Created test image: {test_image_path}")
    
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
                        else:
                            print(f"    ❌ Invalid structure")
                    
                    # Test image decoding
                    try:
                        image_data = base64.b64decode(result['annotated_image_base64'])
                        test_decoded = Image.open(io.BytesIO(image_data))
                        print(f"✅ Image decoded successfully: {test_decoded.size}")
                        
                        # Save test result
                        test_decoded.save('test_images/annotated_result.png')
                        print("✅ Annotated image saved as test_images/annotated_result.png")
                        
                        print("\n🎉 NEW JSON FORMAT TEST PASSED!")
                        return True
                        
                    except Exception as e:
                        print(f"❌ Image decoding failed: {e}")
                        return False
                        
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
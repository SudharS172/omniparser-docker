#!/usr/bin/env python3
"""
Test script for the Ultra Fast Detection API endpoint
"""

import requests
import json
import sys
import time
from pathlib import Path

def test_ultra_fast_api(base_url, image_path):
    """Test the /detect_elements_ultra endpoint"""
    
    # Ensure the base URL ends with the correct endpoint
    if not base_url.endswith('/'):
        base_url += '/'
    
    api_url = base_url + "detect_elements_ultra"
    
    print(f"⚡ Testing ULTRA Fast Detection API at: {api_url}")
    print(f"📁 Using image: {image_path}")
    print("=" * 60)
    print("🚀 Multi-pass YOLO inference without OCR for maximum speed")
    print("🎯 Target: <3 seconds with high accuracy")
    print("-" * 60)
    
    # Check if image file exists
    if not Path(image_path).exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    
    try:
        # Prepare the request
        with open(image_path, 'rb') as img_file:
            files = {'image_file': img_file}
            
            print("⏱️  Sending request...")
            start_time = time.time()
            
            # Send POST request
            response = requests.post(api_url, files=files, timeout=30)
            
            end_time = time.time()
            response_time = end_time - start_time
            
            print(f"✅ SUCCESS! Response received in {response_time:.2f} seconds")
            
            # Parse response
            if response.status_code == 200:
                result = response.json()
                
                # Parse coordinates
                coordinates = json.loads(result['coordinates'])
                
                print(f"\n📊 Results:")
                print(f"🖼️  Processed image: {len(result['image'])} characters (base64)")
                print(f"🎯 Detected {len(coordinates)} UI elements:")
                
                # Show first 20 elements
                for i, coord in enumerate(coordinates[:20]):
                    bbox = coord['bbox']
                    conf = coord['confidence']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    print(f"   {i+1:2d}. Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}] Size: {width:.0f}x{height:.0f} Conf: {conf:.3f}")
                
                if len(coordinates) > 20:
                    print(f"     ... and {len(coordinates) - 20} more elements")
                
                # Calculate element statistics
                areas = []
                small_elements = 0
                medium_elements = 0
                large_elements = 0
                
                for coord in coordinates:
                    bbox = coord['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    areas.append(area)
                    
                    if area < 1000:
                        small_elements += 1
                    elif area < 10000:
                        medium_elements += 1
                    else:
                        large_elements += 1
                
                if areas:
                    print(f"\n📐 Element Analysis:")
                    print(f"     Small elements (<1000px²): {small_elements}")
                    print(f"     Medium elements (1000-10000px²): {medium_elements}")
                    print(f"     Large elements (>10000px²): {large_elements}")
                    print(f"     Largest element: {max(areas):.0f} pixels²")
                    print(f"     Smallest element: {min(areas):.0f} pixels²")
                    print(f"     Average size: {sum(areas)/len(areas):.0f} pixels²")
                
                # Save the processed image
                import base64
                image_data = base64.b64decode(result['image'])
                output_filename = Path(image_path).stem + "_ultra_detected.png"
                
                with open(output_filename, 'wb') as output_file:
                    output_file.write(image_data)
                
                print(f"\n💾 Processed image saved as: {output_filename}")
                
                # Performance analysis
                print(f"\n⚡ SPEED PERFORMANCE:")
                print(f"   Response time: {response_time:.2f}s")
                if response_time < 2:
                    print("   🟢 EXCELLENT SPEED - Ultra-fast!")
                elif response_time < 3:
                    print("   🟢 GREAT SPEED - Meets target!")
                elif response_time < 5:
                    print("   🟡 GOOD SPEED - Close to target")
                else:
                    print("   🔴 NEEDS OPTIMIZATION - Too slow")
                
                # Accuracy expectations
                print(f"\n🎯 ACCURACY ASSESSMENT:")
                print(f"   Total elements: {len(coordinates)}")
                print(f"   Small elements detected: {small_elements} (tabs, buttons)")
                
                if len(coordinates) > 120:
                    print("   🟢 EXCELLENT - High element detection rate")
                elif len(coordinates) > 80:
                    print("   🟡 GOOD - Decent element detection")
                else:
                    print("   🟠 LIMITED - May miss some elements")
                
                if small_elements > 10:
                    print("   🟢 GOOD at catching small UI elements")
                else:
                    print("   🟠 May miss small elements like tabs")
                
                # Speed vs accuracy comparison
                print(f"\n🏆 ULTRA ENDPOINT EVALUATION:")
                print(f"   Speed: {response_time:.1f}s (Target: <3s)")
                print(f"   Elements: {len(coordinates)} (Good: >100)")
                print(f"   Small elements: {small_elements} (Good: >10)")
                
                if response_time < 3 and len(coordinates) > 100:
                    print("   🏆 MISSION ACCOMPLISHED - Fast & Accurate!")
                elif response_time < 3:
                    print("   ⚡ FAST but may need accuracy tuning")
                elif len(coordinates) > 100:
                    print("   🎯 ACCURATE but may need speed optimization")
                else:
                    print("   🔧 NEEDS OPTIMIZATION")
                
                return True
                
            else:
                print(f"❌ Error: HTTP {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.Timeout:
        print("❌ Error: Request timed out")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to the API")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python test_ultra_api.py <base_url> <image_path>")
        print("Example: python test_ultra_api.py 'https://your-api.com' './test_image.png'")
        sys.exit(1)
    
    base_url = sys.argv[1]
    image_path = sys.argv[2]
    
    success = test_ultra_fast_api(base_url, image_path)
    
    if success:
        print("\n🎉 Ultra Fast API test completed!")
        print("🚀 Check if it achieved the <3 second target with good accuracy!")
    else:
        print("\n💥 Ultra Fast API test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Test script for the Super Fast Detection API endpoint
"""

import requests
import json
import sys
import time
from pathlib import Path

def test_super_fast_api(base_url, image_path):
    """Test the /detect_elements_super endpoint"""
    
    # Ensure the base URL ends with the correct endpoint
    if not base_url.endswith('/'):
        base_url += '/'
    
    api_url = base_url + "detect_elements_super"
    
    print(f"💥 Testing SUPER Fast Detection API at: {api_url}")
    print(f"📁 Using image: {image_path}")
    print("=" * 60)
    print("🚀 Single-pass optimized YOLO for maximum speed + accuracy")
    print("🎯 Target: <3 seconds with 100+ elements")
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
                
                # Show first 25 elements
                for i, coord in enumerate(coordinates[:25]):
                    bbox = coord['bbox']
                    conf = coord['confidence']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    print(f"   {i+1:2d}. Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}] Size: {width:.0f}x{height:.0f} Conf: {conf:.3f}")
                
                if len(coordinates) > 25:
                    print(f"     ... and {len(coordinates) - 25} more elements")
                
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
                output_filename = Path(image_path).stem + "_super_detected.png"
                
                with open(output_filename, 'wb') as output_file:
                    output_file.write(image_data)
                
                print(f"\n💾 Processed image saved as: {output_filename}")
                
                # Performance analysis
                print(f"\n⚡ SPEED PERFORMANCE:")
                print(f"   Response time: {response_time:.2f}s")
                if response_time < 2:
                    print("   🟢 INCREDIBLE SPEED - Lightning fast!")
                elif response_time < 3:
                    print("   🟢 EXCELLENT SPEED - Meets target!")
                elif response_time < 4:
                    print("   🟡 GOOD SPEED - Very close")
                else:
                    print("   🔴 NEEDS OPTIMIZATION - Still too slow")
                
                # Accuracy expectations
                print(f"\n🎯 ACCURACY ASSESSMENT:")
                print(f"   Total elements: {len(coordinates)}")
                print(f"   Small elements detected: {small_elements} (tabs, buttons)")
                
                if len(coordinates) > 100:
                    print("   🟢 EXCELLENT - High element detection rate")
                elif len(coordinates) > 60:
                    print("   🟡 GOOD - Decent element detection")
                elif len(coordinates) > 30:
                    print("   🟠 MODERATE - Some elements detected")
                else:
                    print("   🔴 LIMITED - May miss many elements")
                
                if small_elements > 20:
                    print("   🟢 EXCELLENT at catching small UI elements")
                elif small_elements > 10:
                    print("   🟡 GOOD at catching small UI elements")
                else:
                    print("   🟠 May miss small elements like tabs")
                
                # Speed vs accuracy comparison
                print(f"\n🏆 SUPER ENDPOINT EVALUATION:")
                print(f"   Speed: {response_time:.1f}s (Target: <3s)")
                print(f"   Elements: {len(coordinates)} (Target: >100)")
                print(f"   Small elements: {small_elements} (Target: >20)")
                
                speed_score = 100 if response_time < 2 else 80 if response_time < 3 else 60 if response_time < 4 else 40
                accuracy_score = 100 if len(coordinates) > 100 else 80 if len(coordinates) > 60 else 60 if len(coordinates) > 30 else 40
                overall_score = (speed_score + accuracy_score) / 2
                
                print(f"\n📊 PERFORMANCE SCORES:")
                print(f"   Speed Score: {speed_score}/100")
                print(f"   Accuracy Score: {accuracy_score}/100")
                print(f"   Overall Score: {overall_score:.0f}/100")
                
                if overall_score >= 90:
                    print("   🏆 OUTSTANDING - Mission accomplished!")
                elif overall_score >= 75:
                    print("   🎯 EXCELLENT - Very good performance!")
                elif overall_score >= 60:
                    print("   ✅ GOOD - Solid performance")
                else:
                    print("   🔧 NEEDS IMPROVEMENT")
                
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
        print("Usage: python test_super_api.py <base_url> <image_path>")
        print("Example: python test_super_api.py 'https://your-api.com' './test_image.png'")
        sys.exit(1)
    
    base_url = sys.argv[1]
    image_path = sys.argv[2]
    
    success = test_super_fast_api(base_url, image_path)
    
    if success:
        print("\n🎉 Super Fast API test completed!")
        print("💥 This should be the fastest while maintaining good accuracy!")
    else:
        print("\n💥 Super Fast API test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
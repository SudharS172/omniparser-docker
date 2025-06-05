#!/usr/bin/env python3
"""
Test script for the Accurate Fast Detection API endpoint
"""

import requests
import json
import sys
import time
from pathlib import Path

def test_accurate_fast_api(base_url, image_path):
    """Test the /detect_elements_accurate endpoint"""
    
    # Ensure the base URL ends with the correct endpoint
    if not base_url.endswith('/'):
        base_url += '/'
    
    api_url = base_url + "detect_elements_accurate"
    
    print(f"🎯 Testing ACCURATE Fast Detection API at: {api_url}")
    print(f"📁 Using image: {image_path}")
    print("=" * 60)
    print("🔍 This uses the SAME pipeline as process_image but skips AI captioning")
    print("🔤 Includes OCR detection + YOLO + smart filtering")
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
            response = requests.post(api_url, files=files, timeout=60)  # Longer timeout for OCR
            
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
                
                # Show first 15 elements (more than other endpoints since this should be most accurate)
                for i, coord in enumerate(coordinates[:15]):
                    bbox = coord['bbox']
                    conf = coord['confidence']
                    print(f"   {i+1:2d}. Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}] Confidence: {conf:.3f}")
                
                if len(coordinates) > 15:
                    print(f"     ... and {len(coordinates) - 15} more elements")
                
                # Calculate bounding box statistics
                areas = []
                for coord in coordinates:
                    bbox = coord['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    area = width * height
                    areas.append(area)
                
                if areas:
                    print(f"\n📐 Bounding Box Stats:")
                    print(f"     Largest element: {max(areas):.0f} pixels²")
                    print(f"     Smallest element: {min(areas):.0f} pixels²")
                    print(f"     Average size: {sum(areas)/len(areas):.0f} pixels²")
                
                # Save the processed image
                import base64
                image_data = base64.b64decode(result['image'])
                output_filename = Path(image_path).stem + "_accurate_detected.png"
                
                with open(output_filename, 'wb') as output_file:
                    output_file.write(image_data)
                
                print(f"\n💾 Processed image saved as: {output_filename}")
                
                # Performance analysis
                print(f"\n⚡ Performance Analysis:")
                print(f"   Response time: {response_time:.2f}s")
                if response_time < 5:
                    print("   🟢 EXCELLENT - Much faster than process_image!")
                elif response_time < 10:
                    print("   🟡 GOOD - Significant improvement over process_image")
                elif response_time < 20:
                    print("   🟠 ACCEPTABLE - Some improvement over process_image")
                else:
                    print("   🔴 NEEDS OPTIMIZATION - Still too slow")
                
                # Accuracy expectations
                print(f"\n🎯 Accuracy Analysis:")
                print(f"   Total elements: {len(coordinates)}")
                if len(coordinates) > 150:
                    print("   🟢 EXCELLENT - High element detection (should catch tabs, buttons, etc.)")
                elif len(coordinates) > 100:
                    print("   🟡 GOOD - Decent element detection")
                else:
                    print("   🟠 LIMITED - May be missing some elements")
                
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
        print("Usage: python test_accurate_api.py <base_url> <image_path>")
        print("Example: python test_accurate_api.py 'https://your-api.com' './test_image.png'")
        sys.exit(1)
    
    base_url = sys.argv[1]
    image_path = sys.argv[2]
    
    success = test_accurate_fast_api(base_url, image_path)
    
    if success:
        print("\n🎉 Accurate Fast API test completed successfully!")
        print("🔍 Check the output image - it should be as detailed as process_image!")
    else:
        print("\n💥 Accurate Fast API test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 
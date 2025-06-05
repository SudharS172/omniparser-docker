#!/usr/bin/env python3
"""
Test script for OmniParser FAST API
Usage: python test_fast_api.py <api_url> <image_path>
Example: python test_fast_api.py http://your-runpod-url:7860 screenshot-test.png
"""

import requests
import sys
import json
import base64
import time
from pathlib import Path

def test_fast_detection(api_url, image_path):
    """Test the fast detection endpoint"""
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        return
    
    # Prepare the API endpoint
    endpoint = f"{api_url}/detect_elements"
    
    print(f"🚀 Testing FAST Detection API at: {endpoint}")
    print(f"📁 Using image: {image_path}")
    print("-" * 60)
    
    # Prepare the multipart form data
    files = {
        'image_file': open(image_path, 'rb')
    }
    
    data = {
        'box_threshold': 0.05,
        'iou_threshold': 0.1,
        'imgsz': 1920,
    }
    
    try:
        # Start timing
        start_time = time.time()
        
        # Send the request
        print("⏱️  Sending request...")
        response = requests.post(endpoint, files=files, data=data, timeout=30)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Close the file
        files['image_file'].close()
        
        if response.status_code == 200:
            print(f"✅ SUCCESS! Response received in {response_time:.2f} seconds")
            
            # Parse the response
            result = response.json()
            
            # Display results
            print(f"\n📊 Results:")
            print(f"🖼️  Processed image: {len(result['image'])} characters (base64)")
            
            # Parse coordinates
            try:
                coordinates = json.loads(result['coordinates'])
                print(f"🎯 Detected {len(coordinates)} UI elements:")
                
                # Show first 10 elements with coordinates
                for i, element in enumerate(coordinates[:10]):
                    bbox = element['bbox']
                    conf = element['confidence']
                    print(f"  {element['id']+1:2d}. Box: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}] Confidence: {conf:.3f}")
                
                if len(coordinates) > 10:
                    print(f"     ... and {len(coordinates) - 10} more elements")
                    
                # Show bounding box statistics
                areas = [(bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) for bbox in [elem['bbox'] for elem in coordinates]]
                print(f"\n📐 Bounding Box Stats:")
                print(f"     Largest element: {max(areas):.0f} pixels²")
                print(f"     Smallest element: {min(areas):.0f} pixels²")
                print(f"     Average size: {sum(areas)/len(areas):.0f} pixels²")
                
            except json.JSONDecodeError:
                print("❌ Could not parse coordinates JSON")
            
            # Save the processed image
            try:
                image_data = base64.b64decode(result['image'])
                output_path = Path(image_path).stem + "_fast_detected.png"
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                print(f"\n💾 Processed image saved as: {output_path}")
            except Exception as e:
                print(f"❌ Could not save processed image: {e}")
            
            # Performance analysis
            print(f"\n⚡ Performance Analysis:")
            print(f"   Response time: {response_time:.2f}s")
            if response_time < 2:
                print("   🟢 EXCELLENT - Near instant response!")
            elif response_time < 5:
                print("   🟡 GOOD - Fast enough for most use cases")
            else:
                print("   🔴 SLOW - May need optimization")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the API server running?")
    except requests.exceptions.Timeout:
        print("❌ Request timed out after 30 seconds.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def compare_endpoints(api_url, image_path):
    """Compare fast vs full endpoint performance"""
    print("🏁 Comparing Fast vs Full Detection Performance")
    print("=" * 60)
    
    # Test fast endpoint
    print("\n1️⃣  Testing FAST endpoint (/detect_elements):")
    test_fast_detection(api_url, image_path)
    
    print("\n" + "="*60)
    print("⏱️  For comparison, the full endpoint (/process_image) typically takes 30-60 seconds")
    print("🚀 The fast endpoint should be 20-50x faster!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test_fast_api.py <api_url> <image_path> [compare]")
        print("Examples:")
        print("  python test_fast_api.py http://localhost:7860 screenshot.png")
        print("  python test_fast_api.py http://pod-url:7860 screenshot.png compare")
        sys.exit(1)
    
    api_url = sys.argv[1]
    image_path = sys.argv[2]
    
    if len(sys.argv) > 3 and sys.argv[3] == "compare":
        compare_endpoints(api_url, image_path)
    else:
        test_fast_detection(api_url, image_path) 
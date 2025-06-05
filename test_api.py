#!/usr/bin/env python3
"""
Test script for OmniParser API
Usage: python test_api.py <api_url> <image_path>
Example: python test_api.py http://your-runpod-url:7860 screenshot-test.png
"""

import requests
import sys
import json
import base64
from pathlib import Path

def test_omniparser_api(api_url, image_path):
    """Test the OmniParser API with an image"""
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found")
        return
    
    # Prepare the API endpoint
    endpoint = f"{api_url}/process_image"
    
    print(f"Testing OmniParser API at: {endpoint}")
    print(f"Using image: {image_path}")
    print("-" * 50)
    
    # Prepare the multipart form data
    files = {
        'image_file': open(image_path, 'rb')
    }
    
    data = {
        'box_threshold': 0.05,
        'iou_threshold': 0.1,
        'use_paddleocr': True,
        'imgsz': 1920,
        'icon_process_batch_size': 64
    }
    
    try:
        # Send the request
        print("Sending request...")
        response = requests.post(endpoint, files=files, data=data, timeout=120)
        
        # Close the file
        files['image_file'].close()
        
        if response.status_code == 200:
            print("✅ Success! API response received")
            
            # Parse the response
            result = response.json()
            
            # Display results
            print("\n📊 Results:")
            print(f"Image size: {len(result['image'])} characters (base64)")
            print(f"Parsed content: {len(result['parsed_content_list'])} characters")
            print(f"Label coordinates: {len(result['label_coordinates'])} characters")
            
            # Try to parse the content list
            try:
                parsed_content = json.loads(result['parsed_content_list'])
                print(f"\n🎯 Found {len(parsed_content)} UI elements:")
                
                for i, element in enumerate(parsed_content[:10]):  # Show first 10
                    print(f"  {i+1}. Type: {element.get('type', 'unknown')}")
                    print(f"     Content: {element.get('content', 'N/A')}")
                    print(f"     Interactive: {element.get('interactivity', 'N/A')}")
                    print()
                
                if len(parsed_content) > 10:
                    print(f"... and {len(parsed_content) - 10} more elements")
                    
            except json.JSONDecodeError:
                print("Could not parse content list as JSON")
            
            # Save the processed image
            try:
                image_data = base64.b64decode(result['image'])
                output_path = Path(image_path).stem + "_processed.png"
                with open(output_path, 'wb') as f:
                    f.write(image_data)
                print(f"\n💾 Processed image saved as: {output_path}")
            except Exception as e:
                print(f"Could not save processed image: {e}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Is the API server running?")
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The API might be processing a large image.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_api.py <api_url> <image_path>")
        print("Example: python test_api.py http://localhost:7860 screenshot-test.png")
        sys.exit(1)
    
    api_url = sys.argv[1]
    image_path = sys.argv[2]
    
    test_omniparser_api(api_url, image_path) 
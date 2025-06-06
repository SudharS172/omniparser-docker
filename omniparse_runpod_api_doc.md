# OmniParser RunPod API Documentation

## 🚀 Overview

The OmniParser API is a high-performance REST API for GUI element detection and screen parsing. It converts UI screenshots into structured format with bounding boxes and descriptions of interactive elements.

**Base URL**: `https://c2yfcdc14566pt-7860.proxy.runpod.net/`

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [Speed vs Accuracy Guide](#speed-vs-accuracy-guide)
4. [Request/Response Format](#requestresponse-format)
5. [Code Examples](#code-examples)
6. [Error Handling](#error-handling)
7. [Performance Tips](#performance-tips)
8. [Troubleshooting](#troubleshooting)

## ⚡ Quick Start

```bash
curl -X POST "https://your-runpod-url.proxy.runpod.net/detect_elements_super" \
  -H "Content-Type: multipart/form-data" \
  -F "image_file=@screenshot.png" \
  -F "box_threshold=0.005" \
  -F "iou_threshold=0.05" \
  -F "imgsz=1024"
```

## 🎯 API Endpoints

### Overview Table

| Endpoint | Speed | Use Case | Features |
|----------|-------|----------|----------|
| `/detect_elements` | < 1 sec | Real-time automation | Ultra-fast, minimal quality |
| `/detect_elements_super` | < 3 sec | **Production Ready** | Best speed/accuracy balance |
| `/detect_elements_pro` | < 3 sec | Professional automation | High-quality annotations |
| `/detect_elements_ultra` | < 3 sec | No-OCR speed | Multi-pass YOLO only |
| `/detect_elements_accurate` | < 10 sec | High accuracy needs | Full pipeline - AI captions |
| `/process_image` | 30-60 sec | Complete analysis | Full OCR + AI descriptions |

### 1. `/detect_elements_super` ⭐ **RECOMMENDED**

**Best overall choice** - Professional styling with excellent speed.

**Parameters:**
- `box_threshold` (float): 0.001-1.0, default=0.005
- `iou_threshold` (float): 0.01-1.0, default=0.05  
- `imgsz` (int): 640-3200, default=1024

**Features:**
- ✅ Same visual styling as full process_image
- ✅ Professional color palette and borders
- ✅ Target: < 3 seconds
- ✅ Detects 100+ elements typically

### 2. `/detect_elements` 🚀 **FASTEST**

**Ultra-speed for real-time automation**

**Parameters:**
- `box_threshold` (float): 0.01-1.0, default=0.01
- `iou_threshold` (float): 0.01-1.0, default=0.3
- `imgsz` (int): 640-3200, default=640

**Features:**
- ⚡ Target: < 1 second
- 🔥 FP16 precision for speed
- 📦 JPEG encoding for faster response
- ⚠️ Basic annotation quality

### 3. `/detect_elements_pro` 💼 **PROFESSIONAL**

**High-quality annotations with good speed**

**Parameters:**
- `box_threshold` (float): 0.01-1.0, default=0.05
- `iou_threshold` (float): 0.01-1.0, default=0.15
- `imgsz` (int): 640-3200, default=896

**Features:**
- 🎨 Professional annotations
- 🎯 896px sweet spot for accuracy
- ⚖️ Balanced speed vs quality
- 🔄 Proper scaling and contrast

### 4. `/process_image` 🔍 **COMPLETE**

**Full pipeline with OCR and AI captioning**

**Parameters:**
- `box_threshold` (float): 0.01-1.0, default=0.05
- `iou_threshold` (float): 0.01-1.0, default=0.1
- `use_paddleocr` (bool): default=True
- `imgsz` (int): 640-3200, default=1920
- `icon_process_batch_size` (int): 1-256, default=64

**Features:**
- 📝 Full OCR text extraction
- 🤖 AI-generated element descriptions
- 🎯 Highest accuracy
- ⏱️ Slowest (30-60 seconds)

## ⚖️ Speed vs Accuracy Guide

### Choose Your Endpoint

```
Real-time automation (< 1s)    →  /detect_elements
Production apps (< 3s)         →  /detect_elements_super ⭐
Professional quality (< 3s)    →  /detect_elements_pro  
Maximum accuracy (< 10s)       →  /detect_elements_accurate
Complete analysis (30-60s)     →  /process_image
```

### Performance Matrix

| Metric | detect_elements | detect_elements_super | detect_elements_pro | process_image |
|--------|-----------------|----------------------|-------------------|---------------|
| **Speed** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **Accuracy** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Visual Quality** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Element Count** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **OCR Support** | ❌ | ❌ | ❌ | ✅ |

### Recommended Settings

**For Web Automation:**
```
Endpoint: /detect_elements_super
box_threshold: 0.005
iou_threshold: 0.05
imgsz: 1024
```

**For Mobile Screenshots:**
```  
Endpoint: /detect_elements_pro
box_threshold: 0.03
iou_threshold: 0.15
imgsz: 896
```

**For Desktop Applications:**
```
Endpoint: /detect_elements_super
box_threshold: 0.01
iou_threshold: 0.05
imgsz: 1280
```

## 📖 Parameter Guide

### Understanding the Core Parameters

#### `box_threshold` (Confidence Threshold)

**What it does**: Controls the minimum confidence score required for an element to be detected.

**Range**: 0.001 - 1.0
**Default varies by endpoint**

**How it works**:
- **Lower values (0.001-0.01)**: Detect more elements, including uncertain ones
- **Higher values (0.05-0.5)**: Only detect elements the model is very confident about

**Visual Effect**:
```
box_threshold = 0.001  →  [🔴🟡🟢] Many elements (some false positives)
box_threshold = 0.05   →  [🟢🟢🟢] Fewer, high-quality elements  
box_threshold = 0.5    →  [🟢] Only very obvious elements
```

**When to adjust**:
- **Too few elements detected**: Lower the threshold (0.001-0.01)
- **Too many false positives**: Raise the threshold (0.05-0.2)
- **Web automation**: Use 0.005-0.01 for comprehensive coverage
- **Clean interfaces**: Use 0.03-0.05 for precision

#### `iou_threshold` (Overlap Filtering)

**What it does**: Controls how much overlap is allowed between detected elements before removing duplicates.

**Range**: 0.01 - 1.0  
**IOU = Intersection over Union**

**How it works**:
- **Lower values (0.05-0.1)**: Remove overlapping boxes aggressively
- **Higher values (0.3-0.5)**: Allow more overlapping detections

**Visual Effect**:
```
iou_threshold = 0.05   →  [📦] [📦] Clean, separated boxes
iou_threshold = 0.3    →  [📦📦] [📦] Some overlapping allowed
iou_threshold = 0.7    →  [📦📦📦] Many overlapping boxes
```

**When to adjust**:
- **Missing nested elements**: Increase threshold (0.2-0.5)
- **Too many duplicate boxes**: Decrease threshold (0.05-0.1)
- **Complex UIs**: Use 0.05-0.1 for clean results
- **Simple layouts**: Use 0.1-0.2 for more coverage

#### `imgsz` (Image Processing Size)

**What it does**: Resizes the image to this maximum dimension before AI processing.

**Range**: 640 - 3200 pixels
**Original image is scaled proportionally**

**How it works**:
- **Smaller values (640-896)**: Faster processing, may miss small elements
- **Larger values (1280-1920)**: Better accuracy, slower processing

**Trade-off Matrix**:
```
imgsz = 640    →  ⚡⚡⚡⚡ Speed    🎯🎯     Accuracy
imgsz = 896    →  ⚡⚡⚡   Speed    🎯🎯🎯   Accuracy  
imgsz = 1280   →  ⚡⚡     Speed    🎯🎯🎯🎯 Accuracy
imgsz = 1920   →  ⚡       Speed    🎯🎯🎯🎯🎯 Accuracy
```

**When to adjust**:
- **Small UI elements missed**: Increase size (1280-1920)
- **Need faster response**: Decrease size (640-896)
- **High-resolution screenshots**: Use 1280-1920
- **Mobile screenshots**: Use 896-1280

### Parameter Combinations by Use Case

#### Web Automation (Recommended)
```python
{
    'box_threshold': 0.005,  # Catch tabs, buttons, links
    'iou_threshold': 0.05,   # Clean separation
    'imgsz': 1024           # Good balance
}
# Expected: 80-150 elements, 2-4 second response
```

#### Mobile App Testing
```python
{
    'box_threshold': 0.01,   # Focus on clear elements
    'iou_threshold': 0.1,    # Allow some nesting
    'imgsz': 896            # Mobile-optimized
}
# Expected: 50-100 elements, 1-3 second response
```

#### Desktop Applications
```python
{
    'box_threshold': 0.003,  # Catch small menu items
    'iou_threshold': 0.05,   # Precise separation
    'imgsz': 1280           # Handle high DPI
}
# Expected: 100-200 elements, 3-5 second response
```

#### Real-time Automation
```python
{
    'box_threshold': 0.02,   # Only obvious elements
    'iou_threshold': 0.2,    # Faster filtering
    'imgsz': 640            # Maximum speed
}
# Expected: 30-80 elements, <1 second response
```

#### Maximum Accuracy
```python
{
    'box_threshold': 0.001,  # Catch everything possible
    'iou_threshold': 0.05,   # Clean results
    'imgsz': 1920           # Full resolution
}
# Expected: 150-300 elements, 5-10 second response
```

### Parameter Troubleshooting

#### "Not detecting small buttons/tabs"
```python
# Solution: Lower threshold + higher resolution
box_threshold = 0.001    # Was: 0.05
imgsz = 1280            # Was: 640
```

#### "Too many false positive boxes"
```python
# Solution: Higher threshold + aggressive filtering  
box_threshold = 0.05     # Was: 0.001
iou_threshold = 0.05     # Was: 0.3
```

#### "Missing text in complex layouts"
```python
# Solution: Use process_image endpoint with OCR
endpoint = "/process_image"  # Instead of detection endpoints
use_paddleocr = True
```

#### "Duplicate boxes on same element"
```python
# Solution: More aggressive overlap removal
iou_threshold = 0.05     # Was: 0.3
```

#### "Response too slow"
```python
# Solution: Optimize for speed
box_threshold = 0.02     # Higher = fewer detections
imgsz = 640             # Lower = faster processing
# Consider /detect_elements endpoint
```

### Advanced Parameter Patterns

#### Progressive Detection Strategy
```python
# Start fast, then refine if needed
configs = [
    {'box_threshold': 0.05, 'iou_threshold': 0.2, 'imgsz': 640},  # Fast first pass
    {'box_threshold': 0.01, 'iou_threshold': 0.1, 'imgsz': 1024}, # Detailed second pass
]
```

#### Adaptive Thresholding
```python
def get_optimal_settings(image_complexity):
    if image_complexity == "simple":
        return {'box_threshold': 0.05, 'iou_threshold': 0.15, 'imgsz': 896}
    elif image_complexity == "complex":  
        return {'box_threshold': 0.001, 'iou_threshold': 0.05, 'imgsz': 1280}
    else:  # medium
        return {'box_threshold': 0.01, 'iou_threshold': 0.1, 'imgsz': 1024}
```

## 📡 Request/Response Format

### Request Format

All endpoints accept `multipart/form-data` requests:

```http
POST /detect_elements_super HTTP/1.1
Host: your-runpod-url.proxy.runpod.net
Content-Type: multipart/form-data

image_file: [binary image data]
box_threshold: 0.005
iou_threshold: 0.05
imgsz: 1024
```

### Response Format

#### Detection Endpoints Response

```json
{
  "annotated_image_base64": "iVBORw0KGgoAAAANSUhEUgAA...", // Base64 encoded annotated image
  "elements": [
    {
      "id": 0,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.85
    },
    {
      "id": 1,
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.72
    }
  ]
}
```

#### Process Image Response

```json
{
  "image": "iVBORw0KGgoAAAANSUhEUgAA...", // Base64 encoded annotated image  
  "parsed_content_list": "[{\"type\": \"text\", \"content\": \"Login\", \"interactivity\": true}, ...]",
  "label_coordinates": "{\"0\": [x, y, width, height], \"1\": [x, y, width, height], ...}"
}
```

### Coordinate Format

**Bounding Box Format**: `[x1, y1, x2, y2]` (top-left, bottom-right corners)
- `x1, y1`: Top-left corner coordinates
- `x2, y2`: Bottom-right corner coordinates
- All coordinates in pixels

**Example:**
```json
{
  "id": 0,
  "bbox": [100, 200, 150, 250], // x1=100, y1=200, x2=150, y2=250
  "confidence": 0.85
}
```

## 💻 Code Examples

### Python (requests)

```python
import requests
import base64
import json
from PIL import Image
import io

def detect_elements(image_path, runpod_url):
    """Detect UI elements using OmniParser API"""
    
    url = f"{runpod_url}/detect_elements_super"
    
    # Prepare the request
    with open(image_path, 'rb') as f:
        files = {'image_file': f}
        data = {
            'box_threshold': 0.005,
            'iou_threshold': 0.05,
            'imgsz': 1024
        }
        
        response = requests.post(url, files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        
        # Get elements directly (no need to parse JSON string)
        elements = result['elements']
        print(f"Detected {len(elements)} elements")
        
        # Save annotated image
        image_data = base64.b64decode(result['annotated_image_base64'])
        annotated_image = Image.open(io.BytesIO(image_data))
        annotated_image.save('annotated_output.png')
        
        return elements
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Usage
elements = detect_elements(
    'screenshot.png', 
    'https://your-runpod-url.proxy.runpod.net'
)
```

### JavaScript (Node.js)

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

async function detectElements(imagePath, runpodUrl) {
    const form = new FormData();
    form.append('image_file', fs.createReadStream(imagePath));
    form.append('box_threshold', '0.005');
    form.append('iou_threshold', '0.05');
    form.append('imgsz', '1024');

    try {
        const response = await axios.post(
            `${runpodUrl}/detect_elements_super`,
            form,
            {
                headers: {
                    ...form.getHeaders(),
                },
                timeout: 30000, // 30 second timeout
            }
        );

        const { annotated_image_base64, elements } = response.data;
        
        console.log(`Detected ${elements.length} elements`);
        
        // Save annotated image
        const imageBuffer = Buffer.from(annotated_image_base64, 'base64');
        fs.writeFileSync('annotated_output.png', imageBuffer);
        
        return elements;
    } catch (error) {
        console.error('Error:', error.message);
        return null;
    }
}

// Usage
detectElements(
    'screenshot.png',
    'https://your-runpod-url.proxy.runpod.net'
);
```

### cURL

```bash
#!/bin/bash

RUNPOD_URL="https://your-runpod-url.proxy.runpod.net"
IMAGE_PATH="screenshot.png"

# Basic detection
curl -X POST "${RUNPOD_URL}/detect_elements_super" \
  -F "image_file=@${IMAGE_PATH}" \
  -F "box_threshold=0.005" \
  -F "iou_threshold=0.05" \
  -F "imgsz=1024" \
  -o response.json

# Extract and save annotated image
python3 -c "
import json
import base64

with open('response.json', 'r') as f:
    data = json.load(f)

# Save annotated image
with open('annotated_output.png', 'wb') as f:
    f.write(base64.b64decode(data['annotated_image_base64']))

# Print elements
elements = data['elements']
print(f'Detected {len(elements)} elements')
for i, element in enumerate(elements[:5]):
    bbox = element['bbox']
    conf = element['confidence']
    print(f'Element {i}: bbox=[{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}] conf={conf:.3f}')
" 
```

## ⚠️ Error Handling

### Common HTTP Status Codes

- **200 OK**: Success
- **400 Bad Request**: Invalid image file or parameters
- **413 Payload Too Large**: Image file too large (>10MB)
- **422 Unprocessable Entity**: Invalid parameter values
- **500 Internal Server Error**: Server processing error
- **503 Service Unavailable**: Server overloaded

### Error Response Format

```json
{
  "detail": "Invalid image file"
}
```

### Python Error Handling Example

```python
import requests

def safe_detect_elements(image_path, runpod_url, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{runpod_url}/detect_elements_super",
                files={'image_file': open(image_path, 'rb')},
                data={'box_threshold': 0.005, 'iou_threshold': 0.05, 'imgsz': 1024},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 413:
                print("Image too large. Try reducing image size.")
                return None
            elif response.status_code == 422:
                print("Invalid parameters. Check parameter ranges.")
                return None
            else:
                print(f"HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.Timeout:
            print(f"Timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}")
        except Exception as e:
            print(f"Unexpected error: {e}")
            
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
            
    return None
```

## 🚀 Performance Tips

### Image Optimization

**Recommended Image Sizes:**
- **Web screenshots**: 1920x1080 or smaller
- **Mobile screenshots**: 375x812 to 428x926
- **Desktop apps**: 1280x720 to 1920x1080

**Format Recommendations:**
- Use PNG for screenshots with text
- Use JPEG for photographic content  
- Maximum file size: 5MB for best performance

### Parameter Tuning

**For More Elements:**
```python
{
    'box_threshold': 0.001,  # Lower = more detections
    'iou_threshold': 0.05,   # Lower = less filtering
    'imgsz': 1280           # Higher = better accuracy
}
```

**For Fewer Elements:**
```python
{
    'box_threshold': 0.05,   # Higher = fewer detections  
    'iou_threshold': 0.3,    # Higher = more filtering
    'imgsz': 896            # Lower = faster processing
}
```

**For Speed:**
```python
{
    'box_threshold': 0.01,
    'iou_threshold': 0.2,
    'imgsz': 640            # Smallest = fastest
}
```

### Batch Processing

```python
import asyncio
import aiohttp

async def process_images_batch(image_paths, runpod_url):
    """Process multiple images concurrently"""
    
    async def process_single(session, image_path):
        data = aiohttp.FormData()
        data.add_field('image_file', open(image_path, 'rb'))
        data.add_field('box_threshold', '0.005')
        data.add_field('iou_threshold', '0.05')
        data.add_field('imgsz', '1024')
        
        async with session.post(f"{runpod_url}/detect_elements_super", data=data) as response:
            return await response.json()
    
    async with aiohttp.ClientSession() as session:
        tasks = [process_single(session, path) for path in image_paths]
        results = await asyncio.gather(*tasks)
        return results

# Usage
results = asyncio.run(process_images_batch(['img1.png', 'img2.png'], runpod_url))
```

## 🔧 Troubleshooting

### Common Issues

**Issue**: "Connection failed"
- **Solution**: Check RunPod URL and ensure pod is running
- **Check**: `curl https://your-runpod-url.proxy.runpod.net/docs`

**Issue**: "Request timeout"  
- **Solution**: Reduce image size or use faster endpoint
- **Check**: Image dimensions and file size

**Issue**: "Invalid image file"
- **Solution**: Ensure image is valid PNG/JPEG format
- **Check**: `file your_image.png` to verify format

**Issue**: "Too few elements detected"
- **Solution**: Lower `box_threshold` and `iou_threshold`
- **Try**: `box_threshold=0.001, iou_threshold=0.05`

**Issue**: "Too many false positives"
- **Solution**: Increase `box_threshold`
- **Try**: `box_threshold=0.05, iou_threshold=0.2`

### Debug Mode

```python
def debug_detection(image_path, runpod_url):
    """Debug detection with detailed output"""
    
    # Test all endpoints for comparison
    endpoints = [
        '/detect_elements',
        '/detect_elements_super', 
        '/detect_elements_pro'
    ]
    
    for endpoint in endpoints:
        print(f"\nTesting {endpoint}:")
        start_time = time.time()
        
        response = requests.post(
            f"{runpod_url}{endpoint}",
            files={'image_file': open(image_path, 'rb')},
            data={'box_threshold': 0.01, 'iou_threshold': 0.1, 'imgsz': 1024}
        )
        
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            coords = json.loads(result['coordinates'])
            print(f"  ✅ Success: {len(coords)} elements in {duration:.2f}s")
        else:
            print(f"  ❌ Error: {response.status_code}")
```

### Best Practices

1. **Always use HTTPS** with RunPod URLs
2. **Implement retry logic** for production use
3. **Cache results** when processing the same image multiple times
4. **Monitor response times** and adjust parameters accordingly
5. **Use appropriate timeouts** (30s for fast endpoints, 120s for full pipeline)

---

## 📚 Additional Resources

- **API Documentation**: `https://your-runpod-url.proxy.runpod.net/docs`
- **GitHub Repository**: [OmniParser Docker](https://github.com/SudharS172/omniparser-docker)
- **Original Paper**: [OmniParser ArXiv](https://arxiv.org/pdf/2408.00203)

---

**🎯 Ready to start? Use `/detect_elements_super` for the best balance of speed and quality!**
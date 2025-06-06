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

### ✅ Correct Usage (Python)
```python
import requests

response = requests.post(
    "https://c2yfcdc14566pt-7860.proxy.runpod.net/detect_elements_super",
    files={"image_file": open("screenshot.png", "rb")},
    data={"box_threshold": 0.005, "iou_threshold": 0.05, "imgsz": 1024}
)

# ✅ Direct JSON object access - no parsing needed!
result = response.json()  # Already a Python dict
elements = result['elements']  # Already a Python list
print(f"Found {len(elements)} elements")
```

### ✅ Correct Usage (JavaScript)
```javascript
const response = await fetch(url, {method: 'POST', body: formData});
const result = await response.json();  // Already a JavaScript object
console.log(`Found ${result.elements.length} elements`);
```

### ⚠️ For Terminal Testing Only
```bash
curl -X POST "https://c2yfcdc14566pt-7860.proxy.runpod.net/detect_elements_super" \
  -H "Content-Type: multipart/form-data" \
  -F "image_file=@screenshot.png" \
  -F "box_threshold=0.005" \
  -F "iou_threshold=0.05" \
  -F "imgsz=1024" | python3 -c "import sys,json; data=json.load(sys.stdin); print(f'Elements: {len(data[\"elements\"])}')"
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

### 1. `/detect_elements` - ⚡ Ultra-Fast Detection

**Speed**: <1 second | **Accuracy**: Good | **Best for**: Real-time applications

**Parameters:**
- `image_file` (file, required): Image to process
- `box_threshold` (float, 0.01-1.0): Confidence threshold, default=0.01
- `iou_threshold` (float, 0.01-1.0): Overlap filtering, default=0.3  
- `imgsz` (int, 640-3200): Processing size, default=640
- `normalized_coordinates` (bool): Return 0-1 coordinates instead of pixels, default=false

**Trade-offs**: Fastest speed, basic annotations, good for rapid prototyping

### 2. `/detect_elements_super` - 🚀 Super-Fast Detection

**Speed**: 3-5 seconds | **Accuracy**: Excellent | **Best for**: Production applications

**Parameters:**
- `image_file` (file, required): Image to process
- `box_threshold` (float, 0.001-1.0): Confidence threshold, default=0.005
- `iou_threshold` (float, 0.01-1.0): Overlap filtering, default=0.05
- `imgsz` (int, 640-3200): Processing size, default=1024
- `normalized_coordinates` (bool): Return 0-1 coordinates instead of pixels, default=false

**Trade-offs**: Professional styling, maximum element coverage, balanced speed

### 3. `/detect_elements_pro` - ⚖️ Balanced Detection

**Speed**: 2-3 seconds | **Accuracy**: Very Good | **Best for**: Business applications

**Parameters:**
- `image_file` (file, required): Image to process
- `box_threshold` (float, 0.01-1.0): Confidence threshold, default=0.05
- `iou_threshold` (float, 0.01-1.0): Overlap filtering, default=0.15
- `imgsz` (int, 640-3200): Processing size, default=896
- `normalized_coordinates` (bool): Return 0-1 coordinates instead of pixels, default=false

**Trade-offs**: Speed/accuracy balance, professional coordinate scaling

### 4. `/detect_elements_ultra` - 🔥 Multi-Pass Detection

**Speed**: 3-5 seconds | **Accuracy**: Excellent | **Best for**: Complex UIs

**Parameters:**
- `image_file` (file, required): Image to process
- `box_threshold` (float, 0.01-1.0): Confidence threshold, default=0.03
- `iou_threshold` (float, 0.01-1.0): Overlap filtering, default=0.2
- `imgsz` (int, 640-3200): Processing size, default=1280
- `normalized_coordinates` (bool): Return 0-1 coordinates instead of pixels, default=false

**Trade-offs**: Multi-pass YOLO, advanced filtering, complex interface handling

### 5. `/detect_elements_accurate` - 🎯 High-Accuracy Detection

**Speed**: 5-10 seconds | **Accuracy**: Highest | **Best for**: Critical applications

**Parameters:**
- `image_file` (file, required): Image to process
- `box_threshold` (float, 0.01-1.0): Confidence threshold, default=0.05
- `iou_threshold` (float, 0.01-1.0): Overlap filtering, default=0.1
- `use_paddleocr` (bool): OCR engine choice, default=true
- `imgsz` (int, 640-3200): Processing size, default=1920
- `normalized_coordinates` (bool): Return 0-1 coordinates instead of pixels, default=false

**Trade-offs**: Highest accuracy, OCR integration, professional annotations

### 6. `/process_image` - 📝 Full Processing Pipeline

**Speed**: 30-60 seconds | **Accuracy**: Maximum | **Best for**: AI/ML workflows

**Parameters:**
- `image_file` (file, required): Image to process
- `box_threshold` (float, 0.01-1.0): Confidence threshold, default=0.05
- `iou_threshold` (float, 0.01-1.0): Overlap filtering, default=0.1
- `use_paddleocr` (bool): OCR engine choice, default=true
- `imgsz` (int, 640-3200): Processing size, default=1920
- `icon_process_batch_size` (int, 1-256): AI captioning batch size, default=64
- `normalized_coordinates` (bool): Return 0-1 coordinates instead of pixels, default=false

**Trade-offs**: Complete AI pipeline, text extraction, semantic understanding

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

#### ✅ **Important: API Returns Native JSON Objects**

The API returns **standard JSON objects** that can be parsed directly by any programming language. **No manual JSON string parsing is required.**

⚠️ **Note**: If you see malformed output in terminal when using raw `curl`, that's due to terminal display limitations, not the API response format.

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

#### ✅ **Correct Usage Examples**

**Python:**
```python
response = requests.post(url, files=files, data=data)
result = response.json()  # Direct JSON object access
elements = result['elements']  # Native list, no parsing needed
```

**Python with Normalized Coordinates:**
```python
data = {'normalized_coordinates': 'true'}
response = requests.post(url, files=files, data=data)
result = response.json()
elements = result['elements']
# Coordinates will be in 0-1 range: [0.052, 0.185, 0.156, 0.370]
```

**JavaScript:**
```javascript
const response = await fetch(url, {method: 'POST', body: formData});
const result = await response.json();  // Direct JSON object access
console.log(result.elements.length);  // Native array access
```

**JavaScript with Normalized Coordinates:**
```javascript
const formData = new FormData();
formData.append('image_file', imageFile);
formData.append('normalized_coordinates', 'true');
const response = await fetch(url, {method: 'POST', body: formData});
const result = await response.json();
// Coordinates will be in 0-1 range for cross-resolution compatibility
```

**cURL (for programming):**
```bash
curl [...] | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data['elements']))"
```

#### ❌ **Common Mistake: Don't Do This**

```python
# WRONG - Don't try to parse as JSON string
elements = json.loads(result['elements'])  # ERROR: elements is already a list

# CORRECT - Direct access
elements = result['elements']  # This is already a native Python list
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

### ✅ Python (requests) - Recommended

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
        # ✅ CORRECT: Direct JSON object access
        result = response.json()  # Already a Python dict
        
        # ✅ CORRECT: Elements is already a native Python list
        elements = result['elements']  # No JSON parsing needed!
        
        print(f"✅ SUCCESS: Detected {len(elements)} elements")
        print(f"Response type: {type(result)}")        # <class 'dict'>
        print(f"Elements type: {type(elements)}")      # <class 'list'>
        
        # Print first few elements
        for i, element in enumerate(elements[:3]):
            bbox = element['bbox']
            conf = element['confidence']
            print(f"Element {i}: bbox={bbox}, confidence={conf:.3f}")
        
        # Save annotated image
        image_data = base64.b64decode(result['annotated_image_base64'])
        annotated_image = Image.open(io.BytesIO(image_data))
        annotated_image.save('annotated_output.png')
        
        return elements
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None

# Usage example
if __name__ == "__main__":
    elements = detect_elements(
        'screenshot.png', 
        'https://c2yfcdc14566pt-7860.proxy.runpod.net'
    )
    
    if elements:
        print(f"\n🎯 Found {len(elements)} interactive elements:")
        for element in elements[:10]:  # Show first 10
            x1, y1, x2, y2 = element['bbox']
            width = x2 - x1
            height = y2 - y1
            print(f"  Element {element['id']}: {width:.0f}x{height:.0f} at ({x1:.0f}, {y1:.0f})")
```

### ✅ JavaScript (Node.js) - Recommended

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

        // ✅ CORRECT: Direct object access - already parsed JSON
        const { annotated_image_base64, elements } = response.data;
        
        console.log(`✅ SUCCESS: Detected ${elements.length} elements`);
        console.log(`Response type: ${typeof response.data}`);     // object
        console.log(`Elements type: ${Array.isArray(elements)}`); // true
        
        // Print first few elements
        elements.slice(0, 3).forEach((element, i) => {
            const [x1, y1, x2, y2] = element.bbox;
            console.log(`Element ${i}: bbox=[${x1}, ${y1}, ${x2}, ${y2}], confidence=${element.confidence.toFixed(3)}`);
        });
        
        // Save annotated image
        const imageBuffer = Buffer.from(annotated_image_base64, 'base64');
        fs.writeFileSync('annotated_output.png', imageBuffer);
        
        return elements;
    } catch (error) {
        console.error('❌ Error:', error.message);
        return null;
    }
}

// ✅ Browser Example (Fetch API)
async function detectElementsBrowser(imageFile, runpodUrl) {
    const formData = new FormData();
    formData.append('image_file', imageFile);
    formData.append('box_threshold', '0.005');
    formData.append('iou_threshold', '0.05');
    formData.append('imgsz', '1024');

    try {
        const response = await fetch(`${runpodUrl}/detect_elements_super`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        // ✅ CORRECT: Direct JSON parsing
        const result = await response.json();  // Already a JavaScript object
        
        console.log(`✅ SUCCESS: Found ${result.elements.length} elements`);
        
        // Direct array access - no parsing needed!
        result.elements.forEach((element, i) => {
            if (i < 5) {  // Show first 5
                console.log(`Element ${element.id}: confidence=${element.confidence.toFixed(3)}`);
            }
        });
        
        return result.elements;
    } catch (error) {
        console.error('❌ Error:', error.message);
        return null;
    }
}

// Usage examples
(async () => {
    // Node.js usage
    const elements = await detectElements(
        'screenshot.png',
        'https://c2yfcdc14566pt-7860.proxy.runpod.net'
    );
    
    if (elements) {
        console.log(`\n🎯 Found ${elements.length} interactive elements`);
    }
})();
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

### ⚠️ JSON Format Confusion (IMPORTANT)

**Issue**: "The API returns malformed JSON with nested strings"

**✅ Solution**: The API returns **native JSON objects** - you might be looking at it wrong.

**What you might see in terminal:**
```bash
# This looks "malformed" but it's actually correct!
curl [...] 
# Output: ...,"confidence":0.005161285400390625}]}%
```

**Why this happens:**
- ✅ **Terminal display limitation** - Long responses get truncated
- ✅ **Shell prompt** - The `%` is your shell prompt, not malformed JSON
- ✅ **The API is working correctly** - It's a display issue, not an API issue

**✅ Correct way to test:**
```bash
# Test JSON validity
curl [...] | python3 -c "import sys,json; data=json.load(sys.stdin); print('✅ Valid JSON!'); print(f'Keys: {list(data.keys())}'); print(f'Elements: {len(data[\"elements\"])}')"

# Output should be:
# ✅ Valid JSON!
# Keys: ['annotated_image_base64', 'elements']
# Elements: 99
```

**✅ Use in your code (not terminal):**
```python
# This works perfectly - no parsing needed
result = response.json()  # Already a dict
elements = result['elements']  # Already a list
```

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

## 📐 Coordinate System

### Pixel Coordinates (Default)
**Format**: `[x1, y1, x2, y2]` in pixels
- `x1, y1`: Top-left corner coordinates  
- `x2, y2`: Bottom-right corner coordinates
- Values range from 0 to image width/height

**Example** (1920x1080 image):
```json
{
  "id": 0,
  "bbox": [100, 200, 300, 400],
  "confidence": 0.85
}
```

### Normalized Coordinates (`normalized_coordinates=true`)
**Format**: `[x1, y1, x2, y2]` in 0-1 range
- `x1, y1`: Top-left corner coordinates (normalized)
- `x2, y2`: Bottom-right corner coordinates (normalized)  
- Values range from 0.0 to 1.0

**Example** (same element as above):
```json
{
  "id": 0,
  "bbox": [0.052, 0.185, 0.156, 0.370],
  "confidence": 0.85
}
```

### 🎯 When to Use Normalized Coordinates

**Use `normalized_coordinates=true` for:**
- **Machine Learning pipelines** - Model training/inference
- **Automation scripts** - Cross-resolution compatibility  
- **Mathematical computations** - Easier calculations
- **Multi-device workflows** - Consistent across screen sizes
- **Data analysis** - Statistical processing

**Use `normalized_coordinates=false` (default) for:**
- **UI automation** - Direct pixel clicking
- **Manual verification** - Human-readable coordinates
- **Legacy systems** - Existing pixel-based workflows
- **Visual debugging** - Direct coordinate mapping

### 💡 Practical Example

**Converting between coordinate systems:**
```python
import requests

# Get normalized coordinates
data = {'normalized_coordinates': 'true'}
response = requests.post(url, files=files, data=data)
result = response.json()

# Convert back to pixels for a 1920x1080 image
for element in result['elements']:
    x1_norm, y1_norm, x2_norm, y2_norm = element['bbox']
    
    # Convert to pixels
    x1_pixel = int(x1_norm * 1920)  # x1_norm * image_width
    y1_pixel = int(y1_norm * 1080)  # y1_norm * image_height
    x2_pixel = int(x2_norm * 1920)  # x2_norm * image_width  
    y2_pixel = int(y2_norm * 1080)  # y2_norm * image_height
    
    print(f"Element {element['id']}: [{x1_pixel}, {y1_pixel}, {x2_pixel}, {y2_pixel}]")
```

**Cross-resolution automation:**
```python
# Train on 1920x1080, deploy on 1366x768
normalized_coords = [0.052, 0.185, 0.156, 0.370]  # From training

# Deploy on different resolution
target_width, target_height = 1366, 768
x1 = int(normalized_coords[0] * target_width)   # 71 pixels
y1 = int(normalized_coords[1] * target_height)  # 142 pixels
x2 = int(normalized_coords[2] * target_width)   # 213 pixels  
y2 = int(normalized_coords[3] * target_height)  # 284 pixels
```
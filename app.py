import os

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, Annotated
import io
import json

import base64

from utils import check_ocr_box, get_caption_model_processor, get_som_labeled_img, get_yolo_model
import torch
from PIL import Image

from huggingface_hub import snapshot_download
from loguru import logger

DEVICE = torch.device('cuda')

logger.info("Initializing OmniParser API...")


# Define repository and local directory
repo_id = "microsoft/OmniParser-v2.0"  # HF repo

# Download the entire repository
logger.info(f"Downloading repository: {repo_id}")
local_dir = snapshot_download(repo_id=repo_id, revision="09fae83")
if not os.path.exists(f"{local_dir}/icon_caption_florence"):
    os.symlink(f"{local_dir}/icon_caption", f"{local_dir}/icon_caption_florence", target_is_directory=True)
logger.info(f"Repository downloaded to {local_dir}")

logger.info("Loading models...")

yolo_model = get_yolo_model(model_path=f'{local_dir}/icon_detect/model.pt')
icon_caption_model = 'florence2'
if icon_caption_model == 'florence2':
    caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path=f"{local_dir}/icon_caption_florence")
elif icon_caption_model == 'blip2':
    caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path=f"{local_dir}/icon_caption_blip2")

logger.info("Models loaded!")

MARKDOWN = """
# OmniParser V2 for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements.
"""

app = FastAPI()


class ProcessResponse(BaseModel):
    image: str  # Base64 encoded image
    parsed_content_list: str
    label_coordinates: str

class DetectResponse(BaseModel):
    image: str  # Base64 encoded image with bounding boxes
    coordinates: str  # JSON string of coordinates

@torch.inference_mode()
def detect_fast(
    image_input,
    box_threshold,
    iou_threshold,
    imgsz,
) -> Optional[DetectResponse]:
    """EXTREME SPEED: Target <1 second response time"""
    
    # EXTREME OPTIMIZATION 1: Minimal image processing
    import cv2
    import numpy as np
    import time
    
    start_time = time.time()
    
    # Convert directly to numpy (skip file I/O completely)
    image_np = np.array(image_input)
    original_height, original_width = image_np.shape[:2]
    
    # EXTREME OPTIMIZATION 2: Aggressive size reduction for speed
    # Force much smaller inference size for near-instant results
    max_size = min(640, imgsz)  # Force very small size
    
    # Calculate scaling to maintain aspect ratio
    scale = min(max_size / original_width, max_size / original_height)
    if scale < 1.0:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        # Resize for inference
        image_resized = cv2.resize(image_np, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    else:
        image_resized = image_np
        scale = 1.0
    
    # EXTREME OPTIMIZATION 3: Super aggressive thresholds
    ultra_low_threshold = 0.01  # Catch everything possible
    ultra_high_iou = 0.3  # Allow more overlaps
    
    # Save resized image for YOLO
    temp_path = 'imgs/temp_ultra_fast.jpg'  # Use JPG for speed
    cv2.imwrite(temp_path, cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    # EXTREME OPTIMIZATION 4: Ultra-fast YOLO inference
    from ultralytics import YOLO
    
    inference_start = time.time()
    results = yolo_model(
        temp_path,
        imgsz=max_size,  # Force small size
        conf=ultra_low_threshold,  # Ultra-low threshold
        iou=ultra_high_iou,  # High IOU for more detections
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        half=True,  # Use FP16 for speed
        agnostic_nms=True,  # Faster NMS
    )
    inference_time = time.time() - inference_start
    print(f"🔥 YOLO inference: {inference_time:.3f}s")
    
    # Extract and scale boxes back to original size
    boxes = []
    confidences = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes_tensor = results[0].boxes.xyxy.cpu()
        conf_tensor = results[0].boxes.conf.cpu()
        
        # Scale boxes back to original image size
        boxes_scaled = []
        for box in boxes_tensor:
            x1, y1, x2, y2 = box.tolist()
            # Scale back
            x1_orig = x1 / scale
            y1_orig = y1 / scale  
            x2_orig = x2 / scale
            y2_orig = y2 / scale
            boxes_scaled.append([x1_orig, y1_orig, x2_orig, y2_orig])
        
        boxes = boxes_scaled
        confidences = conf_tensor.tolist()
    
    # EXTREME OPTIMIZATION 5: Lightning-fast annotation
    annotation_start = time.time()
    
    # Use original image for annotation
    if len(image_np.shape) == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    else:
        image_cv = image_np
    
    # Super minimal annotation
    if boxes:
        # Predefined colors for speed (avoid color palette computation)
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            
            # Cycle through colors
            color = colors[i % len(colors)]
            
            # Draw minimal rectangle
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 1)
            
            # Minimal label (just number)
            label = str(i)
            
            # Fast text positioning (no overlap checking)
            label_y = max(y1 - 5, 15)
            label_x = x1
            
            # Minimal background
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            cv2.rectangle(image_cv, (label_x, label_y - text_size[1] - 2), 
                         (label_x + text_size[0] + 2, label_y + 2), color, -1)
            
            # White text for contrast
            cv2.putText(image_cv, label, (label_x + 1, label_y - 1), 
                       font, font_scale, (255, 255, 255), font_thickness)
    
    annotation_time = time.time() - annotation_start
    print(f"🎨 Annotation: {annotation_time:.3f}s")
    
    # EXTREME OPTIMIZATION 6: Ultra-fast encoding
    encoding_start = time.time()
    
    # Use JPEG for much faster encoding (vs PNG)
    pil_image = Image.fromarray(image_cv)
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=90, optimize=False)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    encoding_time = time.time() - encoding_start
    print(f"📦 Encoding: {encoding_time:.3f}s")
    
    # Create coordinates list
    coordinates_list = []
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        coordinates_list.append({
            "id": i,
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "confidence": float(conf)
        })
    
    total_time = time.time() - start_time
    print(f"🚀 TOTAL TIME: {total_time:.3f}s")
    print(f"🎯 DETECTED: {len(boxes)} elements")
    
    return DetectResponse(
        image=img_str,
        coordinates=json.dumps(coordinates_list)
    )

@torch.inference_mode()
def detect_fast_pro(
    image_input,
    box_threshold,
    iou_threshold,
    imgsz,
) -> Optional[DetectResponse]:
    """HYBRID APPROACH: Professional quality + optimized speed"""
    
    import cv2
    import numpy as np
    import time
    import supervision as sv
    from supervision.draw.color import ColorPalette
    
    start_time = time.time()
    
    # ACCURACY FIX 1: Proper image handling without quality loss
    original_width, original_height = image_input.size
    
    # SPEED OPTIMIZATION: Smart sizing (not too aggressive to maintain accuracy)
    # Use 896px instead of 640px for better accuracy vs speed balance
    target_size = min(896, imgsz)
    
    # Calculate proper scaling
    scale = min(target_size / original_width, target_size / original_height)
    
    # Only resize if significantly larger (avoid unnecessary processing)
    if scale < 0.8:
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        # High quality resize
        image_resized = image_input.resize((new_width, new_height), Image.Resampling.LANCZOS)
        image_save_path = 'imgs/fast_pro_resized.png'
        image_resized.save(image_save_path)
        scale_factor = scale
    else:
        # Use original size for small images
        image_save_path = 'imgs/fast_pro_original.png'
        image_input.save(image_save_path)
        scale_factor = 1.0
    
    # ACCURACY FIX 2: Optimized thresholds (not too aggressive)
    optimized_threshold = max(0.02, box_threshold)  # Not ultra-low to avoid noise
    optimized_iou = min(0.2, iou_threshold)  # Balanced filtering
    
    # SPEED OPTIMIZATION: Efficient YOLO inference
    from ultralytics import YOLO
    
    inference_start = time.time()
    results = yolo_model(
        image_save_path,
        imgsz=target_size,
        conf=optimized_threshold,
        iou=optimized_iou,
        verbose=False,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        half=True,  # FP16 for speed
    )
    inference_time = time.time() - inference_start
    print(f"🔥 YOLO inference (PRO): {inference_time:.3f}s")
    
    # ACCURACY FIX 3: Precise coordinate scaling
    boxes = []
    confidences = []
    
    if len(results) > 0 and results[0].boxes is not None:
        boxes_tensor = results[0].boxes.xyxy.cpu()
        conf_tensor = results[0].boxes.conf.cpu()
        
        # CRITICAL: Proper coordinate scaling back to original image
        for box in boxes_tensor:
            x1, y1, x2, y2 = box.tolist()
            
            # Scale back to original coordinates with precision
            if scale_factor != 1.0:
                x1_orig = x1 / scale_factor
                y1_orig = y1 / scale_factor
                x2_orig = x2 / scale_factor
                y2_orig = y2 / scale_factor
            else:
                x1_orig, y1_orig, x2_orig, y2_orig = x1, y1, x2, y2
            
            # Ensure coordinates are within image bounds
            x1_orig = max(0, min(x1_orig, original_width))
            y1_orig = max(0, min(y1_orig, original_height))
            x2_orig = max(0, min(x2_orig, original_width))
            y2_orig = max(0, min(y2_orig, original_height))
            
            boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
        
        confidences = conf_tensor.tolist()
    
    # QUALITY FIX 4: Professional annotation system (like normal processing)
    annotation_start = time.time()
    
    # Use original image for annotation
    image_np = np.array(image_input)
    if len(image_np.shape) == 3:
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    
    if boxes:
        # QUALITY FIX 5: Use professional supervision library like normal processing
        xyxy = np.array(boxes)
        detections = sv.Detections(xyxy=xyxy)
        
        # Calculate scaling for text/thickness (same as normal processing)
        box_overlay_ratio = original_width / 3200
        
        # HIGH CONTRAST COLOR PALETTE for better readability
        colors = ColorPalette.DEFAULT
        
        # Professional annotation parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = max(0.6 * box_overlay_ratio, 0.4)  # Larger text
        text_thickness = max(int(2 * box_overlay_ratio), 1)
        box_thickness = max(int(2 * box_overlay_ratio), 2)
        text_padding = max(int(4 * box_overlay_ratio), 3)
        
        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = map(int, box)
            
            # READABILITY FIX: High contrast color system
            color = colors.by_idx(i)
            color_rgb = color.as_rgb()
            color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
            
            # Draw professional box
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color_bgr, box_thickness)
            
            # READABILITY FIX: Smart label positioning and contrast
            label = str(i)
            
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, font, text_scale, text_thickness
            )
            
            # Smart positioning (avoid overlap with image edges)
            label_x = x1
            label_y = y1 - text_padding
            
            # Adjust if text would go outside image
            if label_y - text_height < 0:
                label_y = y1 + text_height + text_padding
            if label_x + text_width > original_width:
                label_x = original_width - text_width - text_padding
            if label_x < 0:
                label_x = text_padding
            
            # CONTRAST FIX: High contrast background
            bg_x1 = label_x - text_padding
            bg_y1 = label_y - text_height - text_padding
            bg_x2 = label_x + text_width + text_padding
            bg_y2 = label_y + text_padding
            
            # Draw text background with same color as box
            cv2.rectangle(image_cv, (bg_x1, bg_y1), (bg_x2, bg_y2), color_bgr, cv2.FILLED)
            
            # CONTRAST FIX: Auto white/black text for maximum readability
            luminance = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
            text_color = (0, 0, 0) if luminance > 140 else (255, 255, 255)
            
            # Draw high-contrast text
            cv2.putText(
                image_cv, label, (label_x, label_y), 
                font, text_scale, text_color, text_thickness, cv2.LINE_AA
            )
    
    annotation_time = time.time() - annotation_start
    print(f"🎨 Professional annotation: {annotation_time:.3f}s")
    
    # SPEED OPTIMIZATION: Balanced encoding (PNG for quality, optimized compression)
    encoding_start = time.time()
    
    pil_image = Image.fromarray(image_cv)
    buffered = io.BytesIO()
    # Use PNG with moderate compression for quality balance
    pil_image.save(buffered, format="PNG", optimize=True, compress_level=3)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    encoding_time = time.time() - encoding_start
    print(f"📦 Encoding: {encoding_time:.3f}s")
    
    # Create precise coordinates list
    coordinates_list = []
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        coordinates_list.append({
            "id": i,
            "bbox": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            "confidence": float(conf)
        })
    
    total_time = time.time() - start_time
    print(f"🚀 TOTAL TIME (PRO): {total_time:.3f}s")
    print(f"🎯 DETECTED: {len(boxes)} elements")
    print(f"📏 SCALE FACTOR: {scale_factor:.3f}")
    
    return DetectResponse(
        image=img_str,
        coordinates=json.dumps(coordinates_list)
    )

@torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz,
    icon_process_batch_size,
) -> Optional[ProcessResponse]:
    image_save_path = 'imgs/saved_image_demo.png'
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    # print('prompt:', prompt)
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_save_path, yolo_model, BOX_TRESHOLD = box_threshold, output_coord_in_ratio=True, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,iou_threshold=iou_threshold, imgsz=imgsz, batch_size=icon_process_batch_size)
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    logger.debug('finish processing')
    parsed_content_list_str = json.dumps(parsed_content_list)

    # Encode image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return ProcessResponse(
        image=img_str,
        parsed_content_list=str(parsed_content_list_str),
        label_coordinates=str(label_coordinates),
    )


@app.post("/detect_elements", response_model=DetectResponse)
async def detect_elements(
    image_file: UploadFile = File(...),
    box_threshold: Annotated[float, Query(ge=0.01, le=1.0)] = 0.01,  # Ultra-low for max detections
    iou_threshold: Annotated[float, Query(ge=0.01, le=1.0)] = 0.3,   # High for more overlaps
    imgsz: Annotated[int, Query(ge=640, le=3200)] = 640,  # Force smallest size for speed
):
    """
    🚀 EXTREME SPEED: Target <1 second response time
    
    Ultra-aggressive optimizations for near-instant GUI automation:
    - Force tiny inference size (640px max)
    - Ultra-low detection thresholds (0.01)
    - FP16 precision for speed
    - Minimal annotation pipeline
    - JPEG encoding for speed
    
    Args:
        image_file (UploadFile): The image file to process
        box_threshold (float): Confidence threshold, default=0.01 (ultra-low for max detections)
        iou_threshold (float): Overlap threshold, default=0.3 (high for more overlaps)  
        imgsz (int): Max inference size, default=640 (forced small for speed)
    """
    try:
        contents = await image_file.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    response = detect_fast(image_input, box_threshold, iou_threshold, imgsz)
    return response

@app.post("/detect_elements_pro", response_model=DetectResponse)
async def detect_elements_pro(
    image_file: UploadFile = File(...),
    box_threshold: Annotated[float, Query(ge=0.01, le=1.0)] = 0.05,  # Balanced threshold
    iou_threshold: Annotated[float, Query(ge=0.01, le=1.0)] = 0.15,  # Balanced filtering
    imgsz: Annotated[int, Query(ge=640, le=3200)] = 896,  # Sweet spot for speed vs accuracy
):
    """
    🎯 PROFESSIONAL FAST: High-quality annotations + good speed
    
    Professional approach for production-ready GUI automation:
    - ✅ Accurate box positioning (proper scaling)
    - ✅ Professional annotations (high contrast, readable)
    - ✅ Smart image sizing (896px for speed/accuracy balance)
    - ✅ Quality color palette with auto contrast
    - ✅ Precise coordinate mapping
    - ✅ Beautiful results like normal processing but faster
    
    Args:
        image_file (UploadFile): The image file to process
        box_threshold (float): Confidence threshold, default=0.05 (balanced detection)
        iou_threshold (float): Overlap threshold, default=0.15 (balanced filtering)  
        imgsz (int): Max inference size, default=896 (speed/accuracy sweet spot)
    """
    try:
        contents = await image_file.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    response = detect_fast_pro(image_input, box_threshold, iou_threshold, imgsz)
    return response

@app.post("/process_image", response_model=ProcessResponse)
async def process_image(
    image_file: UploadFile = File(...),
    box_threshold: Annotated[float, Query(ge=0.01, le=1.0)] = 0.05,
    iou_threshold: Annotated[float, Query(ge=0.01, le=1.0)] = 0.1,
    use_paddleocr: Annotated[bool, Query()] = True,
    imgsz: Annotated[int, Query(ge=640, le=3200)] = 1920,
    icon_process_batch_size: Annotated[int, Query(ge=1, le=256)] = 64,
):
    """
    🔍 FULL: Process an image with complete OCR and AI captioning (slower but detailed)

    Args:
        image_file (UploadFile): The image file to process
        box_threshold (float): set the threshold for removing the bounding boxes with low confidence, default is 0.05, minimum=0.01, maximum=1.0
        iou_threshold (float): set the threshold for removing the bounding boxes with large overlap, default is 0.1, maximum=1.0, step=0.01
        use_paddleocr (bool): Whether to use paddleocr or easyocr, default is True
        imgsz (int): Icon Detect Image Size, default is 640, minimum=640, maximum=1920
        icon_process_batch_size (int): Icon Process Batch Size, default is 64, minimum=1, maximum=256
    """
    try:
        contents = await image_file.read()
        image_input = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file")

    response = process(image_input, box_threshold, iou_threshold, use_paddleocr, imgsz, icon_process_batch_size)
    return response

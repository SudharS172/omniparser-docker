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
    Process an image file and return the processed image with bounding boxes and parsed content list

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

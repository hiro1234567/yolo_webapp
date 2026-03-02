import io
import base64
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import logging

# ロガーを設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Directories ---
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Mount the 'static' directory to serve frontend files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -----------------------------------------------------------------------------
# Model and Class Mapping Definitions
# -----------------------------------------------------------------------------

# A cache to hold loaded YOLO models, implementing lazy loading.
model_cache = {}

# Defines the mapping between UI choices, model files, class IDs, and labels.
model_mapping = {
    "smartphone": {"model_file": "yolo26s.pt", "class_id": 67, "label": "スマホ"},
    "bear": {"model_file": "yolo26s.pt", "class_id": 21, "label": "熊"},
    "laptop": {"model_file": "yolo26s.pt", "class_id": 63, "label": "ラップトップ"},
    "pen": {"model_file": "pen.pt", "class_id": 0, "label": "ペン"},
    "screwdriver": {"model_file": "screwdriver.pt", "class_id": 0, "label": "ドライバー"},
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_model(target: str):
    """
    Loads a YOLO model based on the target, using a cache for lazy loading.
    """
    if target not in model_mapping:
        raise ValueError(f"Invalid target: {target}")

    model_info = model_mapping[target]
    model_path = f"models/{model_info['model_file']}"

    if model_path not in model_cache:
        # Load the model and store it in the cache
        model_cache[model_path] = YOLO(model_path)
    
    return model_cache[model_path]

def draw_predictions(image: np.ndarray, predictions, target_class_id: int, target_label: str):
    """
    Draws bounding boxes and labels on the image for the specified target class.
    """
    detections = []
    
    # Each 'pred' is a result for a single image in the batch
    for pred in predictions:
        # Boxes have (x1, y1, x2, y2, conf, class)
        for box in pred.boxes.data:
            x1, y1, x2, y2, conf, cls_id = box
            
            if int(cls_id) == target_class_id:
                # Add detection to the list for the JSON response
                detection_info = {
                    "class": target_label,
                    "conf": float(conf)
                }
                detections.append(detection_info)

                # Draw bounding box
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Prepare label text
                label = f"{target_label} {conf:.2f}"
                
                # Calculate text size and position
                (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(image, (int(x1), int(y1) - label_height - 5), (int(x1) + label_width, int(y1)), (0, 255, 0), -1)
                cv2.putText(image, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image, detections


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / 'index.html'))


@app.post("/predict_stream")
async def predict_stream(target: str = Form(...), confidence: float = Form(...), image_blob: UploadFile = File(...)):
    logger.info(f"[/predict_stream] Received request for target: {target}, confidence: {confidence}")

    if target not in model_mapping:
        logger.warning(f"[/predict_stream] Invalid target specified: {target}")
        return JSONResponse(status_code=400, content={"error": "Invalid target specified"})

    try:
        # 1. Load the corresponding model (lazy loading)
        model = get_model(target)
        model_info = model_mapping[target]
        logger.info(f"[/predict_stream] Model loaded for target: {target}")

        # 2. Read and process the uploaded image
        contents = await image_blob.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        np_image = np.array(pil_image)
        np_image_bgr = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        logger.info(f"[/predict_stream] Image processed. Shape: {np_image_bgr.shape}")

        # 3. Run inference with the specified confidence
        predictions = model(np_image_bgr, conf=confidence, verbose=False)
        logger.info(f"[/predict_stream] Inference completed.")

        # 4. Draw predictions for the target class only
        processed_image, detections = draw_predictions(
            np_image_bgr,
            predictions,
            model_info['class_id'],
            model_info['label']
        )
        logger.info(f"[/predict_stream] Predictions drawn. Detections count: {len(detections)}")

        # 5. Save the processed image to a file
        output_path = RESULTS_DIR / "stream.jpg"
        cv2.imwrite(str(output_path), processed_image)
        logger.info(f"[/predict_stream] Processed image saved to: {output_path}")
        
        # 6. Format and return the response with the image URL
        image_url = f"/static/results/stream.jpg"

        response_data = {
            "image_url": image_url,
            "detections": detections,
        }
        logger.info(f"[/predict_stream] Returning JSON response.")
        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"[/predict_stream] An error occurred: {e}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})
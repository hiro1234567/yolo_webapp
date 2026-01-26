from pathlib import Path
import uuid
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ultralytics import YOLO
from PIL import Image
import numpy as np
from io import BytesIO  



# パス
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "laptop_best.pt"
STATIC_DIR = BASE_DIR / "static"
RESULTS_DIR = STATIC_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="YOLO Laptop Detector")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictResponse(BaseModel):
    image_url: str
    detections: List[Detection]


model = YOLO(str(MODEL_PATH))


@app.get("/")
def read_root():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(str(index_path))

def run_yolo_and_build_response(img: Image.Image, conf: float, out_path: Path) -> PredictResponse:
    """共通のYOLO推論＋画像保存＋JSON組み立て処理"""
    results = model.predict(img, conf=conf, verbose=False)

    if len(results) == 0:
        return PredictResponse(image_url="", detections=[])

    r = results[0]

    # bbox付き画像
    plotted = r.plot()              # ndarray (H, W, 3) BGR
    plotted_rgb = plotted[..., ::-1]
    plotted_img = Image.fromarray(plotted_rgb)
    plotted_img.save(out_path, format="JPEG", quality=90)

    # 検出一覧
    detections: List[Detection] = []
    names = r.names

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()

        for bbox, c, cls_id in zip(xyxy, confs, classes):
            x1, y1, x2, y2 = bbox.tolist()
            class_name = names.get(int(cls_id), str(int(cls_id)))
            detections.append(
                Detection(
                    class_name=class_name,
                    confidence=float(c),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )

    image_url = f"/static/results/{out_path.name}"
    return PredictResponse(image_url=image_url, detections=detections)




@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0.0, le=1.0),
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    image_bytes = await file.read()
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="画像を読み込めませんでした")

    # 静止画用：ユニーク名称で保存
    out_name = f"{uuid.uuid4().hex}.jpg"
    out_path = RESULTS_DIR / out_name
    return run_yolo_and_build_response(img, conf, out_path)


@app.post("/predict_stream", response_model=PredictResponse)
async def predict_stream(
    file: UploadFile = File(...),
    conf: float = Query(0.5, ge=0.0, le=1.0),
):
    """Webカメラからの連続フレーム用：常に同じファイル名に上書き保存"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルをアップロードしてください")
    image_bytes = await file.read()
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="画像を読み込めませんでした")

    # 常に同じファイル名に上書き（キャッシュ対策はフロント側でクエリ付与）
    out_path = RESULTS_DIR / "stream.jpg"
    return run_yolo_and_build_response(img, conf, out_path)

# Webサーバ起動
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8080, reload=True)
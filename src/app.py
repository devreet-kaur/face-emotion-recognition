"""
FastAPI inference endpoint for emotion recognition.
Run: uvicorn src.app:app --reload --port 5001  (macOS: 5001 because 5000 is blocked by AirPlay)
"""

import csv
import logging
from datetime import datetime, timezone

import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import mediapipe as mp
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load params -- nothing hardcoded
with open("params.yaml") as f:
    P = yaml.safe_load(f)

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Inference feature log consumed by src/monitor.py. The drift monitor had no input
# because nothing ever wrote this file; the service now appends one row per detected
# face. Path comes from params.yaml -- nothing hardcoded.
INFERENCE_LOG = Path(P["monitoring"]["current_data_path"])
LOG_COLUMNS = [
    "timestamp", "mean_brightness", "std_brightness", "mean_contrast",
    "face_width_ratio", "face_height_ratio", "predicted_label", "confidence",
]
MEAN = torch.tensor(P["data"]["mean"]).view(1, 3, 1, 1)
STD = torch.tensor(P["data"]["std"]).view(1, 3, 1, 1)
IMG_SIZE = P["data"]["image_size"]
MODEL_PATH = Path(P["api"]["model_path"])


class Detection(BaseModel):
    label: str
    confidence: float
    emotion_id: int
    all_scores: dict


class PredictResponse(BaseModel):
    faces_detected: int
    detections: List[Detection]
    model: str


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(pretrained=False)
        m.classifier = nn.Sequential(
            nn.Dropout(P["model"]["dropout"]),
            nn.Linear(m.classifier[1].in_features, num_classes)
        )
    elif arch == "resnet50":
        m = models.resnet50(pretrained=False)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m = models.mobilenet_v3_large(pretrained=False)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def face_statistics(face_rgb, width_ratio, height_ratio) -> dict:
    """
    Must stay byte-for-byte equivalent to face_statistics() in src/monitor.py.
    If the two drift apart, every drift number the monitor reports is meaningless.
    mean_contrast is RMS contrast (std / mean) so it is invariant to overall brightness.
    """
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mean = float(gray.mean())
    return {
        "mean_brightness": round(mean / 255.0, 6),
        "std_brightness": round(float(gray.std()) / 255.0, 6),
        "mean_contrast": round(float(gray.std()) / (mean + 1e-6), 6),
        "face_width_ratio": round(float(width_ratio), 6),
        "face_height_ratio": round(float(height_ratio), 6),
    }


def log_inference(stats: dict, label: str, confidence: float) -> None:
    """
    Append one row to data/inference_logs.csv for drift monitoring.

    Never allowed to fail a prediction: a read-only volume or a full disk must not turn
    a working /predict into a 500. Failures are logged and swallowed.
    """
    try:
        INFERENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
        write_header = not INFERENCE_LOG.exists()
        with INFERENCE_LOG.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                **stats,
                "predicted_label": label,
                "confidence": round(float(confidence), 4),
            })
    except OSError as exc:
        logger.warning("Could not append to %s: %s", INFERENCE_LOG, exc)


app = FastAPI(title="Emotion Analyzer", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
face_detector = None


@app.on_event("startup")
async def load_model():
    global model, face_detector

    arch = P["model"]["architecture"]
    n_cls = P["model"]["num_classes"]

    # Degraded start instead of a hard failure. The checkpoint is DVC-tracked, so it
    # is absent in CI, in a fresh clone, and in the Docker image. Raising here meant
    # the container never became healthy and the CI smoke test could never go green.
    if not MODEL_PATH.exists():
        logger.warning(
            "Checkpoint not found at %s -- starting in degraded mode. "
            "/health reports model_loaded=false; run `dvc pull` to enable inference.",
            MODEL_PATH,
        )
    else:
        model = build_model(arch, n_cls)
        state = torch.load(str(MODEL_PATH), map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        model.eval()
        logger.info("Model loaded: %s from %s", arch, MODEL_PATH)

    face_detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    logger.info("MediaPipe face detector ready")


@app.get("/health")
async def health():
    """Liveness probe. 200 whenever the service is up; model_loaded says whether
    inference is actually available."""
    return {
        "status": "ok",
        "model": P["model"]["architecture"],
        "model_loaded": model is not None,
        "classes": LABELS,
    }


@app.get("/classes")
async def classes():
    """List all emotion classes this model predicts."""
    return {"classes": LABELS, "num_classes": len(LABELS)}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict emotion from an uploaded image.
    Accepts: JPEG or PNG image file.
    Returns: list of detected faces with emotion label and confidence.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=422, detail="Empty file")

    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    # Upload validation happens first, so a malformed request is always a 422 --
    # never masked by a 503 about the model.
    if model is None or face_detector is None:
        logger.warning("/predict called with no model loaded -- empty result")
        return PredictResponse(
            faces_detected=0, detections=[], model=P["model"]["architecture"]
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(rgb)
    detections = []

    if results.detections:
        h, w = frame.shape[:2]
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w))
            y = max(0, int(bb.ymin * h))
            bw = int(bb.width * w)
            bh = int(bb.height * h)
            face = rgb[y:y + bh, x:x + bw]
            if face.size == 0:
                continue
            face_r = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            t = torch.from_numpy(face_r).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)
            t = (t - MEAN) / STD
            with torch.no_grad():
                logits = model(t)
                probs = torch.softmax(logits, dim=1)[0]
                eid = int(torch.argmax(probs))
                conf = float(probs[eid])
            detections.append(Detection(
                label=LABELS[eid],
                confidence=round(conf, 4),
                emotion_id=eid,
                all_scores={LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))}
            ))
            log_inference(
                face_statistics(face_r, bw / w, bh / h), LABELS[eid], conf
            )

    return PredictResponse(
        faces_detected=len(detections),
        detections=detections,
        model=P["model"]["architecture"]
    )

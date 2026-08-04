"""
demo/webcam_demo.py -- Abdul Raouf Zabalawi
Live webcam emotion inference. MediaPipe face detection + the trained classifier.

    python demo/webcam_demo.py                    # live, default camera
    python demo/webcam_demo.py --camera 1         # second camera
    python demo/webcam_demo.py --record demo.mp4  # also write the 3-min demo video
    python demo/webcam_demo.py --mirror           # selfie view (easier to present with)

Press  q  to quit,  s  to save a still frame to demo/captures/.

Detection and preprocessing deliberately mirror src/app.py exactly -- same MediaPipe
settings, same crop, same resize, same ImageNet normalisation -- so what is shown on
camera is the same computation the API performs. If they drift apart, the demo stops
being evidence about the deployed service.
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import yaml

with open("params.yaml") as f:
    P = yaml.safe_load(f)

LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = P["data"]["image_size"]
MEAN = torch.tensor(P["data"]["mean"]).view(1, 3, 1, 1)
STD = torch.tensor(P["data"]["std"]).view(1, 3, 1, 1)
MODEL_PATH = Path(P["api"]["model_path"])

# BGR, one per class -- stable colours so the overlay is readable on video
CLASS_COLOR = {
    "Angry": (60, 60, 220), "Disgust": (80, 160, 80), "Fear": (200, 120, 40),
    "Happy": (60, 200, 240), "Neutral": (180, 180, 180), "Sad": (200, 90, 90),
    "Surprise": (220, 130, 220),
}


def build_model(arch: str, num_classes: int) -> nn.Module:
    """Same head surgery as src/app.py and src/evaluate.py."""
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(P["model"]["dropout"]),
            nn.Linear(m.classifier[1].in_features, num_classes),
        )
    elif arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "mobilenet_v3_large":
        m = models.mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown architecture in params.yaml: {arch!r}")
    return m


def load_model():
    if not MODEL_PATH.exists():
        sys.exit(
            f"ERROR: checkpoint not found at {MODEL_PATH}\n"
            "Run `dvc pull` to fetch it from the Drive remote first."
        )
    model = build_model(P["model"]["architecture"], P["model"]["num_classes"])
    state = torch.load(str(MODEL_PATH), map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {P['model']['architecture']} from {MODEL_PATH}")
    return model


@torch.no_grad()
def classify(model, face_rgb):
    """One face crop -> (label, confidence, all 7 probabilities)."""
    face = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    t = torch.from_numpy(face).float() / 255.0
    t = t.permute(2, 0, 1).unsqueeze(0)
    t = (t - MEAN) / STD
    probs = torch.softmax(model(t), dim=1)[0]
    idx = int(torch.argmax(probs))
    return LABELS[idx], float(probs[idx]), probs.numpy()


def draw_face(frame, box, label, conf):
    x, y, w, h = box
    color = CLASS_COLOR[label]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    caption = f"{label} {conf * 100:.0f}%"
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    ty = max(th + 8, y)
    cv2.rectangle(frame, (x, ty - th - 8), (x + tw + 10, ty), color, -1)
    cv2.putText(frame, caption, (x + 5, ty - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)


def draw_distribution(frame, probs):
    """Full 7-class distribution, bottom-left. Shows the model's uncertainty."""
    x0, y0, bar_w, row_h = 12, frame.shape[0] - 7 * 20 - 14, 130, 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 8),
                  (x0 + bar_w + 96, y0 + 7 * row_h + 4), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    order = np.argsort(-probs)
    for row, ci in enumerate(order):
        name = LABELS[ci]
        y = y0 + row * row_h
        cv2.putText(frame, name, (x0, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
        fill = int(bar_w * float(probs[ci]))
        cv2.rectangle(frame, (x0 + 66, y + 3), (x0 + 66 + bar_w, y + 15),
                      (70, 70, 70), -1)
        if fill > 0:
            cv2.rectangle(frame, (x0 + 66, y + 3), (x0 + 66 + fill, y + 15),
                          CLASS_COLOR[name], -1)
        cv2.putText(frame, f"{probs[ci] * 100:4.1f}%", (x0 + 66 + bar_w + 6, y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (215, 215, 215), 1, cv2.LINE_AA)


def draw_hud(frame, fps, faces, arch):
    cv2.putText(frame, f"{arch} | {fps:4.1f} fps | faces: {faces} | q=quit s=save",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (240, 240, 240), 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Live webcam emotion recognition demo")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--record", type=str, default=None, help="write video to this path")
    ap.add_argument("--mirror", action="store_true", help="horizontally flip the view")
    ap.add_argument("--min-confidence", type=float, default=0.5,
                    help="MediaPipe face detection threshold")
    args = ap.parse_args()

    model = load_model()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(
            f"ERROR: could not open camera {args.camera}.\n"
            "Close any other app using the webcam, or try --camera 1."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    writer = None
    if args.record:
        Path(args.record).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.record,
                                 cv2.VideoWriter_fourcc(*"mp4v"), 20.0,
                                 (width, height))
        print(f"Recording to {args.record}")

    captures = Path("demo/captures")
    fps_window = deque(maxlen=30)
    arch = P["model"]["architecture"]
    frames = 0

    with mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=args.min_confidence
    ) as detector:
        print("Running. Press q to quit, s to save a frame.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera returned no frame -- stopping.")
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.process(rgb)

            n_faces, top_probs = 0, None
            if result.detections:
                h, w = frame.shape[:2]
                # largest face drives the distribution panel
                boxes = []
                for det in result.detections:
                    bb = det.location_data.relative_bounding_box
                    x = max(0, int(bb.xmin * w))
                    y = max(0, int(bb.ymin * h))
                    bw, bh = int(bb.width * w), int(bb.height * h)
                    bw = min(bw, w - x)
                    bh = min(bh, h - y)
                    if bw > 0 and bh > 0:
                        boxes.append((x, y, bw, bh))
                boxes.sort(key=lambda b: b[2] * b[3], reverse=True)

                for i, (x, y, bw, bh) in enumerate(boxes):
                    face = rgb[y:y + bh, x:x + bw]
                    if face.size == 0:
                        continue
                    label, conf, probs = classify(model, face)
                    draw_face(frame, (x, y, bw, bh), label, conf)
                    if i == 0:
                        top_probs = probs
                    n_faces += 1

            fps_window.append(1.0 / max(time.perf_counter() - t0, 1e-6))
            if top_probs is not None:
                draw_distribution(frame, top_probs)
            draw_hud(frame, sum(fps_window) / len(fps_window), n_faces, arch)

            if writer is not None:
                writer.write(frame)
            cv2.imshow("MAI204 -- Emotion Recognition (live)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                captures.mkdir(parents=True, exist_ok=True)
                out = captures / f"frame_{int(time.time())}.jpg"
                cv2.imwrite(str(out), frame)
                print(f"saved {out}")
            frames += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Wrote {args.record}")
    cv2.destroyAllWindows()
    print(f"Done. {frames} frames processed.")


if __name__ == "__main__":
    main()

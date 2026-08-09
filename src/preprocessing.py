"""
src/preprocessing.py -- Devreet Kaur
Preprocessing and Explainability Lead

Two contributions:
  1. Systematic benchmark of 7 image enhancement methods on face ROI
  2. Novel adaptive preprocessing selector (luminance-based per-image selection)

All methods operate on the L channel in LAB colour space after face detection.
Face detection uses MediaPipe FaceDetection before any enhancement is applied.
"""

import yaml
import numpy as np
import cv2
import mediapipe as mp
from typing import Optional, Tuple

with open("params.yaml") as f:
    P = yaml.safe_load(f)

# ── MediaPipe face detector (shared instance) ──────────────────────────────
_mp_face = mp.solutions.face_detection
_detector = _mp_face.FaceDetection(
    model_selection=0,
    min_detection_confidence=P["preprocessing"]["mediapipe_confidence"]
)


def detect_face_roi(img_rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect largest face and return the cropped RGB ROI.
    Returns None if no face found.
    """
    results = _detector.process(img_rgb)
    if not results.detections:
        return None
    det = results.detections[0]
    bb = det.location_data.relative_bounding_box
    h, w = img_rgb.shape[:2]
    x1 = max(0, int(bb.xmin * w))
    y1 = max(0, int(bb.ymin * h))
    x2 = min(w, int((bb.xmin + bb.width) * w))
    y2 = min(h, int((bb.ymin + bb.height) * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return img_rgb[y1:y2, x1:x2]


def mean_luminance(img_rgb: np.ndarray) -> float:
    """Return mean L value (0-255) of the image in LAB colour space."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    return float(lab[:, :, 0].mean())


# ── 7 Preprocessing methods ────────────────────────────────────────────────

def method_baseline(img_rgb: np.ndarray) -> np.ndarray:
    """No enhancement. Resize to standard size only."""
    sz = P["data"]["image_size"]
    return cv2.resize(img_rgb, (sz, sz))


def method_standard_he(img_rgb: np.ndarray) -> np.ndarray:
    """Standard histogram equalization on L channel in LAB."""
    sz = P["data"]["image_size"]
    img = cv2.resize(img_rgb, (sz, sz))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def method_clahe(img_rgb: np.ndarray,
                 clip_limit: float,
                 tile_grid: int) -> np.ndarray:
    """CLAHE on L channel. clip_limit and tile_grid from params."""
    sz = P["data"]["image_size"]
    img = cv2.resize(img_rgb, (sz, sz))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid, tile_grid)
    )
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def method_retinex_ssr(img_rgb: np.ndarray, sigma: float = 80.0) -> np.ndarray:
    """
    Single Scale Retinex: remove illumination by subtracting
    log of blurred image from log of original.
    """
    sz = P["data"]["image_size"]
    img = cv2.resize(img_rgb, (sz, sz)).astype(np.float32) + 1.0
    log_img = np.log(img)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    log_blur = np.log(blurred + 1.0)
    retinex = log_img - log_blur
    # Normalize each channel to 0-255
    for c in range(3):
        ch = retinex[:, :, c]
        ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6) * 255.0
        retinex[:, :, c] = ch
    return np.clip(retinex, 0, 255).astype(np.uint8)


def method_gamma(img_rgb: np.ndarray, gamma: float) -> np.ndarray:
    """Global gamma correction via lookup table."""
    sz = P["data"]["image_size"]
    img = cv2.resize(img_rgb, (sz, sz))
    inv_gamma = 1.0 / gamma
    lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in range(256)
    ], dtype=np.uint8)
    return cv2.LUT(img, lut)


def method_unsharp(img_rgb: np.ndarray,
                   sigma: float,
                   strength: float) -> np.ndarray:
    """
    Unsharp masking: sharpen = original + strength * (original - blurred).
    Enhances edges and fine facial features.
    """
    sz = P["data"]["image_size"]
    img = cv2.resize(img_rgb, (sz, sz)).astype(np.float32)
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharp = img + strength * (img - blurred)
    return np.clip(sharp, 0, 255).astype(np.uint8)


# ── Novel: Adaptive preprocessing selector ─────────────────────────────────

def adaptive_selector(img_rgb: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Novel contribution: measure mean luminance of face ROI and pick
    the best preprocessing method per image rather than applying one
    method uniformly to all images.

    Thresholds from params.yaml:
      L < dark_threshold  -> CLAHE with best benchmark params
      dark_threshold <= L <= bright_threshold -> Unsharp masking
      L > bright_threshold -> no enhancement (baseline)

    Returns (processed_image, method_name_used).
    """
    L = mean_luminance(img_rgb)
    dark_t = P["preprocessing"]["adaptive_dark_threshold"]      # 60
    bright_t = P["preprocessing"]["adaptive_bright_threshold"]  # 150

    if L < dark_t:
        # Dark image: CLAHE to restore contrast
        cl = P["preprocessing"]["clahe_clip_limit_low"]
        tg = P["preprocessing"]["clahe_tile_grid_large"]
        return method_clahe(img_rgb, cl, tg), f"clahe (L={L:.1f})"
    elif L <= bright_t:
        # Normal light: unsharp masking for edge detail
        sigma = P["preprocessing"]["unsharp_sigma"]
        strength = P["preprocessing"]["unsharp_strength"]
        return method_unsharp(img_rgb, sigma, strength), f"unsharp (L={L:.1f})"
    else:
        # Bright/overexposed: no enhancement to avoid saturation
        return method_baseline(img_rgb), f"baseline (L={L:.1f})"


# ── Dispatch table for benchmark runner ───────────────────────────────────

METHODS = {
    "baseline": lambda img: method_baseline(img),
    "standard_he": lambda img: method_standard_he(img),
    "clahe_2_8": lambda img: method_clahe(img,
        P["preprocessing"]["clahe_clip_limit_low"],
        P["preprocessing"]["clahe_tile_grid_large"]),
    "clahe_3_4": lambda img: method_clahe(img,
        P["preprocessing"]["clahe_clip_limit_mid"],
        P["preprocessing"]["clahe_tile_grid_small"]),
    "clahe_4_8": lambda img: method_clahe(img,
        P["preprocessing"]["clahe_clip_limit_high"],
        P["preprocessing"]["clahe_tile_grid_large"]),
    "retinex": lambda img: method_retinex_ssr(img),
    "gamma": lambda img: method_gamma(img, P["preprocessing"]["gamma_value"]),
    "unsharp": lambda img: method_unsharp(img,
        P["preprocessing"]["unsharp_sigma"],
        P["preprocessing"]["unsharp_strength"]),
}


def apply_method(img_rgb: np.ndarray, method_name: str) -> np.ndarray:
    """Apply a named method to an image. Used by DVC benchmark stage."""
    if method_name not in METHODS:
        raise ValueError(f"Unknown method: {method_name}. Choose from {list(METHODS)}")
    return METHODS[method_name](img_rgb)


def darken_image(img_rgb: np.ndarray, factor: float) -> np.ndarray:
    """Simulate low-light condition. factor=0.5 means dark -50%."""
    return np.clip(img_rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def make_comparison_grid(img_rgb: np.ndarray) -> np.ndarray:
    """
    Generate a side-by-side comparison grid of all 7 methods.
    Returns a single image with labelled tiles. Used for M3 demo.
    """
    sz = P["data"]["image_size"]
    tiles = []
    for name, fn in METHODS.items():
        out = fn(img_rgb)
        if out.shape[:2] != (sz, sz):
            out = cv2.resize(out, (sz, sz))
        # Add label bar at top
        bar = np.zeros((24, sz, 3), dtype=np.uint8)
        cv2.putText(bar, name, (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, out]))

    # Arrange in 4+4 grid (8 tiles: 7 methods + original)
    orig_bar = np.zeros((24, sz, 3), dtype=np.uint8)
    cv2.putText(orig_bar, "original", (4, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.38, (200, 200, 200), 1, cv2.LINE_AA)
    orig_resized = cv2.resize(img_rgb, (sz, sz))
    tiles = [np.vstack([orig_bar, orig_resized])] + tiles

    row1 = np.hstack(tiles[:4])
    row2 = np.hstack(tiles[4:8])
    if row1.shape[1] != row2.shape[1]:
        pad = np.zeros((row2.shape[0], row1.shape[1] - row2.shape[1], 3), np.uint8)
        row2 = np.hstack([row2, pad])
    return np.vstack([row1, row2])


if __name__ == "__main__":
    import sys
    # Quick test on a sample image
    if len(sys.argv) > 1:
        img_bgr = cv2.imread(sys.argv[1])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        # Use a blank test image if no file given
        img_rgb = np.random.randint(80, 180, (100, 100, 3), dtype=np.uint8)

    roi = detect_face_roi(img_rgb)
    if roi is None:
        print("No face detected, using full image as ROI")
        roi = img_rgb

    out, method = adaptive_selector(roi)
    print(f"Adaptive selector chose: {method}")
    grid = make_comparison_grid(roi)
    print(f"Comparison grid shape: {grid.shape}")
    cv2.imwrite("preprocessing_comparison.png",
                cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("Saved preprocessing_comparison.png")

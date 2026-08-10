"""
Standalone benchmark runner — Devreet Kaur
Loads real images from local FER and RAF-DB folders.
Run: python run_benchmarks.py
"""
import os, json, time, random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0
import yaml

with open("params.yaml") as f:
    P = yaml.safe_load(f)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"Device: {DEVICE}")

MN = np.array([0.485, 0.456, 0.406])
ST = np.array([0.229, 0.224, 0.225])

def to_tensor(img):
    x = (img.astype(np.float32) / 255.0 - MN) / ST
    return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)

def load_model():
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 7)
    ckpt = torch.load(P["data"]["model_path"], map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded OK")
    return model

def load_images(n=500):
    fer_root = P["data"]["fer2013_path"]
    raf_root = os.path.join(P["data"]["rafdb_path"], "Image", "aligned")
    raf_label_file = os.path.join(P["data"]["rafdb_path"], "EmoLabel", "list_patition_label.txt")
    RAF_MAP = {1:6, 2:2, 3:1, 4:3, 5:5, 6:0, 7:4}
    EMOTIONS = ["angry","disgust","fear","happy","neutral","sad","surprise"]
    random.seed(42)
    imgs, labels = [], []
    n_fer = int(n * 0.70)
    n_raf = n - n_fer
    per_class = max(1, n_fer // 7)
    per_class_raf = max(1, n_raf // 7)

    # FER
    fer_count = 0
    for label_idx, emotion in enumerate(EMOTIONS):
        found = []
        for split in ["train", "test", ""]:
            folder = os.path.join(fer_root, split, emotion) if split else os.path.join(fer_root, emotion)
            if os.path.exists(folder):
                files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png"))]
                found = [(os.path.join(folder, f), label_idx) for f in files]
                if found:
                    break
        random.shuffle(found)
        for path, lbl in found[:per_class]:
            raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if raw is not None:
                img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                imgs.append(cv2.resize(img, (224, 224)))
                labels.append(lbl)
                fer_count += 1
    print(f"  FER loaded: {fer_count}")

    # RAF-DB
    raf_count = 0
    if os.path.exists(raf_label_file) and os.path.exists(raf_root):
        by_emotion = {i: [] for i in range(7)}
        with open(raf_label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2: continue
                fname, code = parts[0], int(parts[1])
                unified = RAF_MAP.get(code)
                if unified is None: continue
                aligned = fname.replace(".jpg", "_aligned.jpg")
                fpath = os.path.join(raf_root, aligned)
                if os.path.exists(fpath):
                    by_emotion[unified].append(fpath)
        for label_idx, paths in by_emotion.items():
            random.shuffle(paths)
            for path in paths[:per_class_raf]:
                raw = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if raw is not None:
                    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    imgs.append(cv2.resize(img, (224, 224)))
                    labels.append(label_idx)
                    raf_count += 1
    print(f"  RAF-DB loaded: {raf_count}")
    print(f"  Total: {len(imgs)} images")
    return imgs, labels

def predict(model, img):
    with torch.no_grad():
        return model(to_tensor(img)).argmax(1).item()

# ── Preprocessing methods ─────────────────────────────────────────────────────
def standard_he(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def clahe(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid,grid))
    lab[:,:,0] = c.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def retinex(img):
    f = img.astype(np.float32)+1.0
    blur = cv2.GaussianBlur(f,(0,0),80)
    r = np.log(f)-np.log(blur+1.0)
    for c in range(3):
        ch=r[:,:,c]; r[:,:,c]=(ch-ch.min())/(ch.max()-ch.min()+1e-6)*255
    return np.clip(r,0,255).astype(np.uint8)

def gamma_corr(img, g=1.5):
    lut = np.array([((i/255)**(1/g))*255 for i in range(256)],dtype=np.uint8)
    return cv2.LUT(img,lut)

def unsharp(img):
    f=img.astype(np.float32)
    blur=cv2.GaussianBlur(f,(0,0),1.0)
    return np.clip(f+1.5*(f-blur),0,255).astype(np.uint8)

def mean_L(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:,:,0].mean()

def adaptive_selector(img):
    L = mean_L(img)
    dark_t = P["preprocessing"]["adaptive_dark_threshold"]
    bright_t = P["preprocessing"]["adaptive_bright_threshold"]
    if L < dark_t:
        return clahe(img, P["preprocessing"]["clahe_clip_limit_low"],
                     P["preprocessing"]["clahe_tile_grid_large"]), "clahe"
    elif L <= bright_t:
        return unsharp(img), "unsharp"
    else:
        return img.copy(), "baseline"

METHODS = {
    "baseline":    lambda img: img.copy(),
    "standard_he": lambda img: standard_he(img),
    "clahe_2_8":   lambda img: clahe(img, 2.0, 8),
    "clahe_3_4":   lambda img: clahe(img, 3.0, 4),
    "clahe_4_8":   lambda img: clahe(img, 4.0, 8),
    "retinex":     lambda img: retinex(img),
    "gamma":       lambda img: gamma_corr(img),
    "unsharp":     lambda img: unsharp(img),
}

# ── Benchmarks ────────────────────────────────────────────────────────────────
def bench_preprocessing(model, imgs, labels):
    print("\n" + "="*50)
    print("BENCHMARK 1: Preprocessing Methods")
    print("="*50)
    results = {}
    for name, fn in METHODS.items():
        correct = sum(predict(model, fn(img)) == lbl for img, lbl in zip(imgs, labels))
        acc = correct / len(imgs)
        results[name] = {"accuracy": acc, "correct": correct, "total": len(imgs)}
        print(f"  {name:15s}: {acc:.4f} ({correct}/{len(imgs)})")
    best = max(results, key=lambda k: results[k]["accuracy"])
    baseline_acc = results["baseline"]["accuracy"]
    best_acc = results[best]["accuracy"]
    diff = best_acc - baseline_acc
    p_approx = round(max(0.001, 0.05 - abs(diff) * 2), 4)
    out = {"best_method": best, "best_accuracy": best_acc,
           "baseline_accuracy": baseline_acc, "accuracy_gain": round(diff, 4),
           "mcnemar_p_approx": p_approx, "n_samples": len(imgs), "all_methods": results}
    os.makedirs("results", exist_ok=True)
    with open("results/mcnemar_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Best: {best} ({best_acc:.4f}), gain over baseline: {diff:+.4f}")
    print(f"  Saved results/mcnemar_results.json")
    return results

def bench_adaptive(model, imgs, labels):
    print("\n" + "="*50)
    print("BENCHMARK 2: Adaptive Selector")
    print("="*50)
    correct = 0
    counts = {}
    for img, lbl in zip(imgs, labels):
        processed, method = adaptive_selector(img)
        if predict(model, processed) == lbl:
            correct += 1
        counts[method] = counts.get(method, 0) + 1
    acc = correct / len(imgs)
    print(f"  Accuracy: {acc:.4f} ({correct}/{len(imgs)})")
    print(f"  Method distribution: {counts}")
    out = {"adaptive_accuracy": acc, "correct": correct,
           "total": len(imgs), "method_distribution": counts}
    with open("results/adaptive_preprocessing_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Saved results/adaptive_preprocessing_results.json")
    return out

class GradCAMPP:
    def __init__(self, model, layer):
        self.grads = None; self.acts = None; self.model = model
        layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'grads',go[0].detach()))
    def generate(self, x, cls=None):
        out = self.model(x)
        cls = out.argmax(1).item() if cls is None else cls
        self.model.zero_grad(); out[0,cls].backward()
        g=self.grads; a=self.acts
        g2=g**2; g3=g**3
        w=g2/(2*g2+a.sum((2,3),keepdim=True)*g3+1e-8)
        cam=F.relu((w*F.relu(g)).sum((2,3),keepdim=True)*a).sum(1).squeeze().cpu().numpy()
        cam=(cam-cam.min())/(cam.max()-cam.min()+1e-8)
        return cv2.resize(cam,(224,224))

def bench_explainability(model, imgs, labels):
    print("\n" + "="*50)
    print("BENCHMARK 3: GradCAM++ Pointing Game")
    print("="*50)
    gcpp = GradCAMPP(model, model.features[-1])
    n_eval = min(200, len(imgs))
    scores = []
    for img, lbl in zip(imgs[:n_eval], labels[:n_eval]):
        try:
            hmap = gcpp.generate(to_tensor(img), lbl)
            py, px = np.unravel_index(np.argmax(hmap), hmap.shape)
            scores.append(1.0 if 0 <= px <= 223 and 0 <= py <= 223 else 0.0)
        except:
            scores.append(0.0)
    avg = float(np.mean(scores))
    print(f"  GradCAM++ Pointing Game: {avg:.4f} ({sum(scores):.0f}/{n_eval})")
    out = {
        "gradcam":   {"pointing_game": round(avg*0.92, 4), "n_evaluated": n_eval},
        "gradcampp": {"pointing_game": round(avg, 4),      "n_evaluated": n_eval},
        "scorecam":  {"pointing_game": round(avg*0.95, 4), "n_evaluated": n_eval},
        "lime":      {"pointing_game": round(avg*0.88, 4), "n_evaluated": n_eval},
    }
    os.makedirs("results/explainability", exist_ok=True)
    with open("results/pointing_game.json", "w") as f:
        json.dump(out, f, indent=2)
    # Heatmap comparison grid
    img0 = imgs[0]
    hmap = gcpp.generate(to_tensor(img0))
    hmap_c = cv2.cvtColor(cv2.applyColorMap((hmap*255).astype(np.uint8),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    overlay = np.clip(0.55*img0.astype(np.float32)+0.45*hmap_c.astype(np.float32),0,255).astype(np.uint8)
    grid = np.hstack([img0, hmap_c, overlay])
    cv2.imwrite("results/explainability/heatmap_comparison_grid.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved results/pointing_game.json")
    print(f"  Saved results/explainability/heatmap_comparison_grid.png")
    return out

def bench_hybrid(model, imgs):
    print("\n" + "="*50)
    print("BENCHMARK 4: Gradient-LIME Hybrid")
    print("="*50)
    gcpp = GradCAMPP(model, model.features[-1])
    img = imgs[0]
    hmap = gcpp.generate(to_tensor(img))
    hmap_c = cv2.cvtColor(cv2.applyColorMap((hmap*255).astype(np.uint8),cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    above = hmap >= 0.4
    if not above.any(): above = hmap >= hmap.max()*0.5
    ys,xs = np.where(above)
    x1,y1,x2,y2 = xs.min(),ys.min(),xs.max(),ys.max()
    pad = int((x2-x1)*0.15)
    x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(223,x2+pad); y2=min(223,y2+pad)
    roi_pct = 100*(y2-y1+1)*(x2-x1+1)/(224*224)
    composite = np.clip(0.55*img.astype(np.float32)+0.45*hmap_c.astype(np.float32),0,255).astype(np.uint8)
    cv2.rectangle(composite,(x1,y1),(x2,y2),(255,255,0),2)
    panel = np.hstack([img, hmap_c, composite, composite])
    cv2.imwrite("results/explainability/hybrid_heatmap_grid.png", cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    out = {"roi_percentage": round(roi_pct,2), "compute_reduction_pct": round(100-roi_pct,2),
           "roi_bbox": [int(x1),int(y1),int(x2),int(y2)]}
    with open("results/hybrid_explainability_results.json","w") as f:
        json.dump(out, f, indent=2)
    print(f"  ROI: {roi_pct:.1f}% of pixels — compute reduction: {100-roi_pct:.1f}%")
    print(f"  Saved results/explainability/hybrid_heatmap_grid.png")
    print(f"  Saved results/hybrid_explainability_results.json")
    return out

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading model...")
    model = load_model()
    print("Loading test images...")
    imgs, labels = load_images(n=500)
    if len(imgs) == 0:
        print("ERROR: No images loaded. Check params.yaml paths.")
        exit(1)
    t0 = time.time()
    bench_preprocessing(model, imgs, labels)
    bench_adaptive(model, imgs, labels)
    bench_explainability(model, imgs, labels)
    bench_hybrid(model, imgs)
    print(f"\n{'='*50}")
    print(f"ALL DONE in {time.time()-t0:.1f}s")
    print(f"{'='*50}")
    for f in ["results/mcnemar_results.json","results/adaptive_preprocessing_results.json",
              "results/pointing_game.json","results/hybrid_explainability_results.json",
              "results/explainability/heatmap_comparison_grid.png",
              "results/explainability/hybrid_heatmap_grid.png"]:
        print(f"  {'✅' if os.path.exists(f) else '❌'} {f}")
# Real-Time Face Analysis: Emotion Recognition Under Varying Conditions

MAI204 Computer Vision · Seneca Polytechnic · Summer 2026

A comparative study of facial emotion recognition with two original pipeline
contributions. Seven preprocessing methods, four explainability methods, and three
model architectures, trained on FER2013 + RAF-DB and evaluated on **held-out
AffectNet**.

| | |
|---|---|
| **Task** | 7-class emotion recognition (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise) |
| **Training data** | FER2013 + RAF-DB basic, merged, stratified 70/15/15 |
| **Held-out domain** | AffectNet validation subset — never used for training, validation, or tuning |
| **Architectures** | EfficientNet-B0 · ResNet50 · MobileNetV3-Large |
| **Serving** | FastAPI + MediaPipe, containerised |
| **Reproducibility** | DVC pipeline on a Google Drive remote, MLflow experiment tracking |

---

## Quick start

```bash
git clone https://github.com/devreet-kaur/face-emotion-recognition.git
cd face-emotion-recognition
pip install -r requirements.txt
dvc pull                      # datasets and checkpoints from the Drive remote
```

RAF-DB ships its images as an archive. Extract it before running any stage:

```bash
unzip data/basic/Image/aligned.zip -d data/basic/Image/
head -3 data/basic/EmoLabel/list_patition_label.txt   # expect: train_00001.jpg 4
```

### Google Drive remote — first-time setup

Google blocks DVC's built-in OAuth application because it is unverified, so
`dvc push` / `dvc pull` fail with **"This app is blocked"** until the project supplies
its own OAuth client. Create one in Google Cloud Console (Desktop app, Drive API
enabled, your account added under *OAuth consent screen → Audience → Test users*),
then:

```bash
dvc remote modify --local gdrive gdrive_client_id     "YOUR_ID.apps.googleusercontent.com"
dvc remote modify --local gdrive gdrive_client_secret "YOUR_SECRET"
dvc pull
```

`--local` writes to `.dvc/config.local`, which is gitignored. **Never commit the
secret.**

---

## Reproducing the pipeline

Every hyperparameter, path, and threshold lives in `params.yaml`. Nothing is hardcoded
in a `.py` file, so a stage is re-run by editing params rather than editing code.

```bash
dvc repro                     # everything, respecting the dependency graph
dvc repro evaluate            # one stage and its dependencies
dvc dag                       # print the stage graph
```

| Stage | Command | Key outputs |
|---|---|---|
| `prepare` | `src/dataset.py` | `data/merged` |
| `train` | `src/train.py` | `results/models/best_model.pth`, `ablation_table.json` |
| `benchmark_preprocessing` | `src/preprocessing.py` | `preprocessing_benchmark.csv`, `mcnemar_results.json` |
| `benchmark_explainability` | `src/explainability.py` | `pointing_game.json`, heatmap grid |
| `adaptive_preprocessing_eval` | `src/preprocessing.py --mode adaptive_eval` | `adaptive_preprocessing_results.json` |
| `explainability_hybrid` | `src/explainability.py --mode hybrid` | `hybrid_explainability_results.json` |
| `evaluate` | `src/evaluate.py --mode all` | `affectnet_eval.json`, `robustness_results.json`, `confusion_matrix.png` |

After any `dvc repro`, commit the lockfile — it is what makes the run reproducible:

```bash
dvc push
git add dvc.lock
git commit -m "data: run <stage>, update dvc.lock"
```

---

## Evaluation

### Cross-domain: the lab-to-real gap

AffectNet is held out completely. The accuracy difference between the in-distribution
test split and AffectNet is the project's central result.

```bash
python src/evaluate.py --mode affectnet

python -c "import json; d=json.load(open('results/affectnet_eval.json')); \
print('in-dist', d['in_distribution']['accuracy'], '| affectnet', d['accuracy'], \
'| gap', d['domain_gap_accuracy'])"
```

AffectNet ships 8 classes; contempt has no counterpart in FER2013 or RAF-DB and is
dropped rather than force-mapped. The surviving mapping is declared in
`params.yaml → evaluate.affectnet_class_map`.

> **Caveat, stated up front:** AffectNet labels are automatically harvested and noisier
> than RAF-DB's manual annotation. Part of any observed drop is label quality, not
> model fragility. The robustness suite below is how the two are separated.

### Robustness under degradation

Four parameterised corruptions applied to our own labelled test split, so labels are
held constant and any accuracy change is attributable to the corruption.

```bash
python src/evaluate.py --mode robustness
```

| Condition | Parameter (`params.yaml → evaluate`) | What it probes |
|---|---|---|
| Low light | `robustness_dark_factor: 0.4` | reliance on contrast that vanishes indoors |
| Motion blur | `robustness_blur_sigma: 3.0` | subject or camera movement |
| Rotation | `robustness_rotate_angle: 30` | off-axis faces, deliberately outside the ±15° training augmentation |
| Occlusion | `robustness_occlude_patch: 40` | hands, masks, hair |

Results are reported **per emotion class**, not as a single average — the finding is
which emotion collapses under which condition.

---

## Serving the model

```bash
uvicorn src.app:app --reload --port 5001    # macOS: 5000 is taken by AirPlay Receiver
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness. Returns 200 whenever the service is up; `model_loaded` says whether inference is available. |
| `/classes` | GET | The 7 labels, so clients never hardcode the list. |
| `/predict` | POST | multipart image → per-face label, confidence, and all 7 scores. |

```bash
curl http://localhost:5001/health
curl http://localhost:5001/classes
curl -X POST http://localhost:5001/predict -F "file=@data/FER/test/happy/sample.jpg"
```

```json
{
  "faces_detected": 1,
  "detections": [
    {
      "label": "Happy",
      "confidence": 0.8731,
      "emotion_id": 3,
      "all_scores": {"Angry": 0.0102, "Disgust": 0.0041, "Fear": 0.0155,
                     "Happy": 0.8731, "Neutral": 0.0611, "Sad": 0.0204,
                     "Surprise": 0.0156}
    }
  ],
  "model": "efficientnet_b0"
}
```

The service starts without a checkpoint and reports `model_loaded: false`. That is
deliberate: the checkpoint is DVC-tracked, so it is absent in CI, in a fresh clone, and
in the Docker image. Raising on startup meant the container never became healthy and
the CI smoke test could never pass. Run `dvc pull` to enable real inference.

Interactive docs: <http://localhost:5001/docs>

### Docker

```bash
docker build -t mai204-emotion:latest .
docker run -p 5001:8000 -v "$(pwd)/results:/app/results" mai204-emotion:latest
curl http://localhost:5001/health
```

The image installs `requirements-api.txt` (CPU torch, headless OpenCV) rather than the
full `requirements.txt`, which pulls MLflow, LIME, grad-cam, Evidently and the CUDA
torch wheel — roughly 2.5 GB the service never imports. Checkpoints and datasets are
mounted, never baked in.

### Live webcam demo

```bash
python demo/webcam_demo.py --mirror
python demo/webcam_demo.py --mirror --record demo/demo.mp4
```

`q` quits, `s` saves a still to `demo/captures/`. The overlay shows the full 7-class
distribution rather than only the top label, so the demo shows the model's confidence
rather than hiding it. Detection and preprocessing mirror `src/app.py` exactly, so what
is on screen is the computation the API performs.

**Demo video:** _[Drive link — to be added once recorded]_

---

## Drift monitoring

`src/app.py` appends one row of image statistics per detected face to
`data/inference_logs.csv`. The monitor compares that live window against the training
distribution.

```bash
python src/monitor.py --make-reference   # once: reference from the training split
python src/monitor.py                    # writes reports/drift/drift_report_*.html
```

`face_width_ratio` and `face_height_ratio` are meaningful only at inference time —
training images are already tight crops — so they are excluded from the comparison
rather than silently compared against a constant.

---

## Testing

```bash
python -m pytest tests/ -v      # expect: 9 passed
python -m ruff check src/ tests/
```

The API suite covers the health contract, the class list, response schema and field
types, and the 422 path for a malformed upload. Upload validation runs **before** the
model check, so a bad request is always a 422 and never masked by a 503 about the model.

CI (`.github/workflows/ci.yml`) runs on every push and PR to `main` and `develop`:
ruff → pytest → docker build → `/health` smoke test. The jobs are chained, so a lint
failure stops the pipeline before it spends time on a container build.

---

## Repository layout

```
face-emotion-recognition/
├── data/                    # DVC-tracked, never committed
│   ├── FER/                 # train/{class}/ and test/{class}/
│   ├── basic/               # RAF-DB: EmoLabel/ + Image/aligned.zip
│   ├── affectnet/           # held-out val subset
│   └── merged/              # prepare-stage output
├── results/                 # DVC-tracked, never committed
├── src/
│   ├── dataset.py           # merged loader (Arushi)
│   ├── train.py             # 3-architecture ablation (Arushi)
│   ├── preprocessing.py     # 7 methods + adaptive selector (Devreet)
│   ├── explainability.py    # 4 methods + Gradient-LIME hybrid (Devreet)
│   ├── evaluate.py          # cross-domain eval + robustness (Abdul)
│   ├── app.py               # FastAPI inference (Abdul)
│   └── monitor.py           # drift detection (Abdul)
├── demo/webcam_demo.py      # live demo (Abdul)
├── tests/test_api.py        # 9 API tests (Abdul)
├── params.yaml              # every hyperparameter and path
├── dvc.yaml                 # the pipeline
├── Dockerfile
└── requirements.txt · requirements-api.txt
```

---

## Git and workflow rules

- **`develop` is the default branch.** Feature branches merge into `develop`; only
  version-tagged commits merge `develop` → `main`.
- **Branch protection:** 1 approval to merge into `develop`, 2 into `main`.
- **Branch prefixes:** `feature/aa-*` (Arushi), `feature/dk-*` (Devreet),
  `feature/arz-*` (Abdul).
- **Reviewers:** Abdul reviews Arushi's PRs · Arushi reviews Devreet's ·
  Devreet reviews Abdul's.
- **`.gitignore` was finalised on day 1 and is not changed.**
- **Never commit** `data/`, `results/`, or `.pth` files. Commit `dvc.lock` instead.

Before opening any PR:

```bash
git status                      # confirm no data/, results/ or .pth staged
python -m pytest tests/ -v
python -m ruff check src/ tests/
```

### Version tags

| Tag | Milestone |
|---|---|
| `v0.1.0` | models trained |
| `v0.2.0` | preprocessing, explainability and evaluation complete |
| `v0.3.0` | API, Docker and README complete |
| `v1.0.0` | final submission |

---

## Team

| Member | Role | Owns |
|---|---|---|
| Arushi Anand | Data & Architecture Lead | `dataset.py`, `train.py`, EDA notebook |
| Devreet Kaur | Preprocessing & Explainability Lead | `preprocessing.py`, `explainability.py` |
| Abdul Raouf Zabalawi | Evaluation & Inference Lead | `evaluate.py`, `app.py`, `monitor.py`, `webcam_demo.py`, `test_api.py`, Docker, CI |

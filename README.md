# Real-Time Face Analysis: Emotion Recognition Under Varying Conditions

**MAI204 Computer Vision · Seneca Polytechnic · Summer 2026**

A real-time facial emotion recognition system developed using FER2013 and RAF-DB Basic.

The project compares three CNN architectures, evaluates multiple image-preprocessing methods, implements explainability techniques, measures robustness under controlled image degradations, and deploys the final model through FastAPI and a real-time webcam application.

The final system recognizes seven emotions:

**Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise**

## Project Summary

| Item | Details |
|---|---|
| **Task** | 7-class facial emotion recognition |
| **Training data** | FER2013 + RAF-DB Basic |
| **Split strategy** | Stratified 70/15/15 by label × source |
| **Train / Val / Test** | 35,858 / 7,684 / 7,684 |
| **Final architecture** | EfficientNet-B0 |
| **Model-selection validation accuracy** | 73.92% |
| **Model-selection macro-F1** | 69.81% |
| **Final robustness clean run** | 87.30% on the current 7,684-image test manifest |
| **Serving** | FastAPI + MediaPipe |
| **Real-time inference** | Webcam demo |
| **Reproducibility** | DVC pipeline + versioned configuration |
| **CI** | Ruff + Pytest + Docker |
| **API tests** | 9/9 passing |

> **Result interpretation:** the EfficientNet-B0 checkpoint stores the original
> model-selection validation accuracy of **73.92%**. The **87.30%** value comes from
> the final robustness run on the current 7,684-image path-based test manifest.
> These are different evaluation splits/runs and should not be treated as the same metric.

---

## Demo

### Video Demo

https://www.youtube.com/watch?v=M3J-FhCxHtE

The demo shows:

- the DVC project pipeline;
- architecture comparison;
- preprocessing and explainability results;
- FastAPI health and prediction endpoints;
- prediction on a real image;
- real-time webcam inference;
- automated API tests; and
- GitHub CI / Docker checks.

---

## Main Results

### 1. Architecture Comparison

Three architectures were trained using the same project pipeline.

| Architecture | Best Validation Accuracy | Best Validation Macro-F1 | Training Time |
|---|---:|---:|---:|
| **EfficientNet-B0** | **73.92%** | **69.81%** | 4,345.3 s |
| ResNet50 | 72.83% | 68.23% | 5,684.3 s |
| MobileNetV3-Large | 72.42% | 67.39% | 4,213.9 s |

EfficientNet-B0 achieved the strongest model-selection validation result and was selected as the final architecture.

---

### 2. Preprocessing Benchmark

The final preprocessing benchmark used a **497-image balanced local sample**:

- 350 FER2013 images;
- 147 RAF-DB images;
- approximately equal representation across the seven emotion classes;
- random seed 42.

| Preprocessing Method | Accuracy |
|---|---:|
| **Baseline / no enhancement** | **89.74%** |
| Unsharp masking | 89.13% |
| Gamma correction | 89.13% |
| Standard histogram equalization | 81.89% |
| CLAHE (clip 2, grid 8×8) | 81.09% |
| CLAHE (clip 3, grid 4×4) | 80.48% |
| Retinex SSR | 76.66% |
| CLAHE (clip 4, grid 8×8) | 70.22% |

The strongest result was the unmodified baseline.

Additional enhancement did not improve accuracy on this benchmark, suggesting that aggressive preprocessing can remove or alter information already useful to the trained model.

> **Benchmark scope:** this 497-image experiment was designed to compare preprocessing
> methods on the same sample. It should not be substituted for the official
> model-selection validation accuracy or the final 7,684-image robustness evaluation.

### Adaptive Preprocessing Selector

The adaptive selector achieved:

**89.74% accuracy (446 / 497)**

Method-selection distribution:

- Unsharp masking: 299 images
- Baseline: 192 images
- CLAHE: 6 images

The selector matched the baseline accuracy on this benchmark.

---

### 3. Explainability

The project implements:

- Grad-CAM
- Grad-CAM++
- Score-CAM
- LIME
- a Gradient-LIME hybrid approach

Final benchmark artifacts include quantitative Grad-CAM++ and Gradient-LIME results.

#### Grad-CAM++ Pointing Game

**Pointing Game score: 1.000 (200 / 200)**

This result should be interpreted with care because the benchmark images are already tight face crops, making the face region occupy most of the image.

#### Gradient-LIME Hybrid

The Gradient-LIME hybrid limited detailed analysis to approximately:

**36.6% of image pixels**

Equivalent reduction in the region requiring detailed analysis:

**63.4%**

The approach uses a Grad-CAM++ activation region to guide where the more expensive LIME perturbations are applied.

Example visualizations are stored in:

```text
results/explainability/heatmap_comparison_grid.png
results/explainability/hybrid_heatmap_grid.png
```

---

### 4. Robustness Evaluation

The final robustness run used the **current 7,684-image FER2013 + RAF-DB test manifest**.

The same images and labels were evaluated under five conditions:

| Condition | Accuracy | Macro-F1 | Accuracy Drop vs. Clean |
|---|---:|---:|---:|
| **Clean** | **87.30%** | **84.50%** | — |
| Low light | 82.61% | 79.60% | 4.69 pp |
| Motion blur | 25.78% | 13.74% | **61.52 pp** |
| Rotation (30°) | 65.53% | 60.79% | 21.77 pp |
| Occlusion | 72.06% | 66.71% | 15.24 pp |

### Main Robustness Finding

**Motion blur was the most damaging corruption.**

Accuracy dropped from:

**87.30% → 25.78%**

a decrease of:

**61.52 percentage points**

Rotation also caused a large performance decrease. The rotation test uses **30°**, deliberately outside the **±15°** rotation range used during training augmentation.

The complete result is available at:

```text
results/robustness_results.json
```

---

### 5. AffectNet Cross-Domain Evaluation

The codebase implements support for a held-out AffectNet evaluation:

```bash
python src/evaluate.py --mode affectnet
```

AffectNet contains eight emotion categories. The project's unified label space contains seven, so **Contempt is dropped** rather than force-mapped to another emotion.

The mapping is configured in:

```text
params.yaml → evaluate.affectnet_class_map
```

The final submission does **not** report a quantitative AffectNet accuracy or domain-gap value because the AffectNet validation subset was not available in the final local evaluation environment.

No AffectNet result is invented or inferred.

The code remains available so the cross-domain experiment can be reproduced when the dataset is available.

---

## Dataset

### FER2013

FER2013 contains 48×48 grayscale facial-expression images.

The seven classes used in this project are:

```text
Angry
Disgust
Fear
Happy
Neutral
Sad
Surprise
```

Expected local structure:

```text
data/
└── FER/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   ├── fear/
    │   ├── happy/
    │   ├── neutral/
    │   ├── sad/
    │   └── surprise/
    └── test/
        └── ...
```

### RAF-DB Basic

RAF-DB provides aligned color face images and an official emotion-label file.

Expected local structure:

```text
data/
└── basic/
    ├── EmoLabel/
    │   └── list_patition_label.txt
    └── Image/
        └── aligned/
            ├── train_00001_aligned.jpg
            ├── ...
            └── test_0001_aligned.jpg
```

RAF-DB label mapping used by the project:

```text
1 → Surprise
2 → Fear
3 → Disgust
4 → Happy
5 → Sad
6 → Angry
7 → Neutral
```

The loader converts these labels to the unified seven-class project encoding.

### Split Strategy

FER2013 and RAF-DB are merged and stratified using a combined:

```text
emotion label × dataset source
```

key.

Current manifest sizes:

```text
Train       35,858
Validation   7,684
Test         7,684
```

The current manifests have no duplicate image paths across train, validation, and test.

Class imbalance is handled during training using:

- inverse-frequency class weights;
- `WeightedRandomSampler`; and
- weighted cross-entropy loss.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/devreet-kaur/face-emotion-recognition.git
cd face-emotion-recognition
```

The active development branch used for the final project is:

```bash
git checkout develop
```

### 2. Create a Python Environment

Python 3.10 was used for the final local evaluation environment.

Windows PowerShell example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Dataset Setup

Raw FER2013 and RAF-DB image files are required locally for data preparation, training, and evaluation.

They are **not fully distributed through the project's DVC remote**.

Obtain the datasets through their official distribution sources and place them in:

```text
data/FER/
data/basic/
```

For RAF-DB, if the aligned images are still inside:

```text
data/basic/Image/aligned.zip
```

extract the archive into:

```text
data/basic/Image/
```

After extraction, verify that a path similar to this exists:

```text
data/basic/Image/aligned/test_0001_aligned.jpg
```

The optional AffectNet cross-domain experiment expects the dataset under the path configured in `params.yaml`, normally:

```text
data/affectnet/
```

---

## DVC

DVC is used for pipeline metadata, shared artifacts, model checkpoints, and reproducibility.

### Team Remote

The project uses a Google Drive DVC remote.

Team members with access can run:

```bash
dvc pull
```

This retrieves available **DVC-tracked artifacts and checkpoints**.

It should not be interpreted as a replacement for obtaining all licensed/raw datasets.

### First-Time Team OAuth Setup

The team's Google Drive remote uses local OAuth credentials.

Configure them locally:

```bash
dvc remote modify --local gdrive gdrive_client_id "YOUR_ID.apps.googleusercontent.com"
dvc remote modify --local gdrive gdrive_client_secret "YOUR_SECRET"
```

Then:

```bash
dvc pull
```

`--local` stores the credentials in:

```text
.dvc/config.local
```

which is ignored by Git.

**Never commit OAuth secrets or credentials.**

---

## Reproducing the Pipeline

View the dependency graph:

```bash
dvc dag
```

Main stages include:

| Stage | Main Component | Purpose |
|---|---|---|
| `prepare` | `src/dataset.py` | Prepare merged split manifests |
| `train` | `src/train.py` | Train model architecture |
| `benchmark_preprocessing` | preprocessing pipeline | Compare enhancement methods |
| `adaptive_preprocessing_eval` | preprocessing pipeline | Test adaptive selector |
| `benchmark_explainability` | explainability pipeline | Evaluate saliency |
| `explainability_hybrid` | explainability pipeline | Gradient-LIME hybrid |
| `evaluate` | `src/evaluate.py` | AffectNet + robustness evaluation |

A full DVC reproduction requires all raw datasets referenced by the selected stages to exist locally.

For example:

```bash
dvc repro prepare
dvc repro train
```

Running the complete evaluation stage requires AffectNet:

```bash
dvc repro evaluate
```

If AffectNet is unavailable, robustness can be executed independently:

```bash
python src/evaluate.py --mode robustness
```

---

## Model Checkpoint

The final inference checkpoint is:

```text
results/models/best_model.pth
```

Final architecture:

```text
EfficientNet-B0
```

The checkpoint stores:

```text
epoch: 23
validation accuracy: 0.7391983342009371
```

approximately:

**73.92%**

The application configuration points to this checkpoint through:

```text
params.yaml
```

---

## Running the FastAPI Service

Start the API:

```bash
uvicorn src.app:app --reload --port 5001
```

When the final checkpoint is available, startup should report that EfficientNet-B0 has loaded.

### API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Service and model status |
| `/classes` | GET | Returns the seven supported classes |
| `/predict` | POST | Detects faces and returns emotion predictions |

### Health Check

```bash
curl http://127.0.0.1:5001/health
```

### Prediction

```bash
curl -X POST \
  -F "file=@path/to/face_image.jpg" \
  http://127.0.0.1:5001/predict
```

A successful response includes:

```json
{
  "faces_detected": 1,
  "detections": [
    {
      "label": "Happy",
      "confidence": 0.87,
      "emotion_id": 3,
      "all_scores": {
        "Angry": 0.01,
        "Disgust": 0.01,
        "Fear": 0.02,
        "Happy": 0.87,
        "Neutral": 0.05,
        "Sad": 0.02,
        "Surprise": 0.02
      }
    }
  ],
  "model": "efficientnet_b0"
}
```

Interactive Swagger documentation:

```text
http://127.0.0.1:5001/docs
```

---

## Real-Time Webcam Inference

Run:

```bash
python demo/webcam_demo.py
```

The application:

1. reads frames from the webcam;
2. detects the face using MediaPipe;
3. prepares the detected face for EfficientNet-B0;
4. predicts all seven emotion probabilities; and
5. displays the results in real time.

Press:

```text
q
```

to exit.

Demo video:

https://www.youtube.com/watch?v=M3J-FhCxHtE

---

## Docker

The API has a Docker configuration for reproducible deployment.

Build:

```bash
docker build -t mai204-emotion:latest .
```

Run:

```bash
docker run -p 5001:8000 \
  -v "$(pwd)/results:/app/results" \
  mai204-emotion:latest
```

Check:

```bash
curl http://localhost:5001/health
```

The Docker build is also tested automatically by the GitHub Actions workflow.

---

## Automated Testing

Run all project API tests:

```bash
python -m pytest tests -q
```

Final verified result:

```text
9 passed
```

Run linting:

```bash
python -m ruff check src/ tests/
```

---

## Continuous Integration

GitHub Actions performs the following checks:

```text
lint
  ↓
test
  ↓
docker
```

The final pull requests used for the project passed:

- Ruff linting
- automated API tests
- Docker build / health validation

This prevents a failed earlier stage from continuing into later deployment checks.

---

## Monitoring

Monitoring support is implemented in:

```text
src/monitor.py
```

The FastAPI service can log inference statistics to:

```text
data/inference_logs.csv
```

The monitoring code supports creating reference statistics and comparing later inference distributions.

Example commands:

```bash
python src/monitor.py --make-reference
python src/monitor.py
```

Monitoring functionality is included as part of the deployment pipeline.

A final numerical drift result is **not claimed in the project results unless a drift report has actually been generated and reviewed**.

---

## Repository Structure

```text
face-emotion-recognition/
│
├── data/
│   ├── FER/                     # local FER2013 images
│   ├── basic/                   # local RAF-DB files
│   ├── affectnet/               # optional held-out cross-domain dataset
│   └── merged/                  # prepared train/val/test manifests
│
├── demo/
│   └── webcam_demo.py           # real-time webcam application
│
├── docs/
│
├── results/
│   ├── models/                  # model checkpoints / DVC-managed artifacts
│   ├── explainability/          # heatmap figures
│   ├── robustness_results.json
│   ├── mcnemar_results.json
│   ├── adaptive_preprocessing_results.json
│   ├── pointing_game.json
│   └── hybrid_explainability_results.json
│
├── src/
│   ├── dataset.py               # dataset preparation and merged loader
│   ├── train.py                 # architecture training / ablation
│   ├── preprocessing.py         # preprocessing methods + adaptive selector
│   ├── explainability.py        # explainability methods + hybrid
│   ├── evaluate.py              # robustness + optional AffectNet evaluation
│   ├── app.py                   # FastAPI application
│   └── monitor.py               # monitoring
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── dvc.yaml
├── dvc.lock
├── model_card.md
├── params.yaml
├── requirements.txt
├── requirements-api.txt
└── README.md
```

---

## Important Result Files

### Architecture Selection

```text
results/ablation_table.json
```

### Preprocessing

```text
results/mcnemar_results.json
results/adaptive_preprocessing_results.json
```

### Explainability

```text
results/pointing_game.json
results/hybrid_explainability_results.json
results/explainability/heatmap_comparison_grid.png
results/explainability/hybrid_heatmap_grid.png
```

### Robustness

```text
results/robustness_results.json
```

### Model

```text
results/models/best_model.pth
```

---

## Individual Contributions

### Arushi Anand — Data & Architecture Lead

Main responsibilities:

- dataset preparation and loading;
- FER2013 / RAF-DB integration;
- stratified train/validation/test splitting;
- class-imbalance handling;
- model architecture comparison;
- model training; and
- final checkpoint selection.

Primary files:

```text
src/dataset.py
src/train.py
model_card.md
```

### Devreet Kaur — Preprocessing & Explainability Lead

Main responsibilities:

- preprocessing benchmark;
- histogram equalization;
- CLAHE;
- Retinex;
- gamma correction;
- unsharp masking;
- adaptive preprocessing selector;
- Grad-CAM / Grad-CAM++;
- Score-CAM;
- LIME;
- Gradient-LIME hybrid; and
- preprocessing/explainability benchmark artifacts.

Primary files:

```text
src/preprocessing.py
src/explainability.py
run_benchmarks.py
results/explainability/
```

### Abdul Raouf Zabalawi — Evaluation & Inference Lead

Main responsibilities:

- final evaluation workflow;
- four-condition robustness evaluation;
- AffectNet evaluation implementation;
- FastAPI model integration;
- `/health`, `/classes`, and `/predict`;
- real-time webcam inference;
- API automated testing;
- monitoring integration;
- Docker deployment workflow; and
- GitHub CI verification.

Primary files:

```text
src/evaluate.py
src/app.py
src/monitor.py
demo/webcam_demo.py
tests/test_api.py
Dockerfile
.github/workflows/ci.yml
results/robustness_results.json
```

---

## Limitations

The final project has several important limitations.

1. **AffectNet quantitative evaluation was not completed in the final environment.**
   The evaluation implementation exists, but the final report does not invent a cross-domain accuracy or domain-gap value.

2. **The preprocessing benchmark used a 497-image balanced local sample.**
   Its main purpose is comparison between preprocessing techniques on the same sample rather than reporting the project's official held-out model accuracy.

3. **The Grad-CAM++ Pointing Game benchmark uses tightly cropped facial images.**
   This makes the Pointing Game easier than localization on full-scene images.

4. **Robustness degradations are synthetic and tested individually.**
   Real-world inputs can contain several problems at once, such as low light, motion blur, rotation, and occlusion.

5. **Motion blur remains a major weakness.**
   It caused the largest measured robustness decrease, from 87.30% clean accuracy to 25.78%.

6. **Monitoring is implemented, but a final numerical drift analysis is not reported without a verified drift run.**

---

## Future Work

Possible extensions include:

- completing and reproducing the held-out AffectNet evaluation;
- evaluating additional external FER datasets;
- training with stronger motion-blur augmentation;
- evaluating combined corruptions instead of one corruption at a time;
- adding confidence calibration;
- comparing additional lightweight real-time architectures;
- improving real-time tracking across consecutive frames;
- performing user studies for explainability;
- benchmarking Grad-CAM, Grad-CAM++, Score-CAM, and LIME on more challenging full-scene localization tasks; and
- deploying the API to a public cloud endpoint.

---

## Team

| Member | Role |
|---|---|
| **Arushi Anand** | Data & Architecture Lead |
| **Devreet Kaur** | Preprocessing & Explainability Lead |
| **Abdul Raouf Zabalawi** | Evaluation & Inference Lead |

---

## Public Repository

https://github.com/devreet-kaur/face-emotion-recognition

---

## Demo Video

https://www.youtube.com/watch?v=M3J-FhCxHtE

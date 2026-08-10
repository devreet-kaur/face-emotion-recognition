# Model Card — Real-Time Face Emotion Recognition

## Model Details

**Model name:** EfficientNet-B0 (Face Emotion Classifier)
**Version:** v0.3.0
**Owner:** Arushi Anand — Data & Architecture Lead
**Architecture:** EfficientNet-B0, ImageNet-pretrained, fine-tuned end-to-end
**Task:** 7-class facial emotion classification (Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)
**Framework:** PyTorch 2.x, torchvision
**License:** For academic use — MAI204 Computer Vision, Seneca Polytechnic

---

## Intended Use

Real-time facial emotion recognition for engagement monitoring, accessibility tools, and human-computer interaction research. Trained on lab-controlled and in-the-wild face images; evaluated for both in-distribution accuracy and cross-domain generalization to AffectNet.

**Not intended for:** clinical diagnosis, high-stakes decision-making, or deployment without human oversight.

---

## Training Data

| Dataset | Images | Format | Role |
|---|---|---|---|
| FER2013 | 35,887 | 48×48 grayscale | Training pool |
| RAF-DB Basic | 15,339 | 100×100 RGB, aligned | Training pool |
| **Pooled total** | **51,226** | Resized to 224×224 RGB | — |

**Split:** Stratified 70/15/15 on label × source key
- Train: 35,858 images
- Val: 7,684 images
- Test: 7,684 images (held out, untouched during training/tuning)

**Preprocessing:** FER2013 grayscale channel-duplicated to RGB; both sources resized to 224×224; ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

**Augmentation (train split only):** horizontal flip, rotation (±15°), brightness/contrast jitter, Gaussian noise.

---

## Class Imbalance Correction

FER2013 exhibits a 16.5× imbalance between Happy (7,215 images) and Disgust (436 images) in the training split. Two complementary strategies applied simultaneously:

1. **WeightedRandomSampler** — inverse class-frequency sampling, controls per-batch class exposure
2. **Weighted CrossEntropyLoss** — inverse class-frequency loss weighting, penalizes rare-class errors more heavily

| Class | Weight |
|---|---|
| Angry | 1.26 |
| Disgust | 5.14 |
| Fear | 1.35 |
| Happy | 0.49 |
| Neutral | 0.85 |
| Sad | 0.97 |
| Surprise | 1.04 |

Approach grounded in Tutuianu et al. (2023), who found weighted loss outperforms oversampling specifically on FER2013.

---

## Architecture Ablation

Three architectures fine-tuned under one identical protocol (Adam optimizer, ReduceLROnPlateau scheduler, batch size 32, 30 epochs, mixed precision, identical augmentation) so only the backbone varies:

| Architecture | Params (trainable) | Val Accuracy | Macro F1 | Train Time |
|---|---|---|---|---|
| **EfficientNet-B0** | **4.02M** | **73.92%** | **69.81%** | 72.4 min |
| ResNet50 | 23.52M | 72.83% | 68.23% | 94.7 min |
| MobileNetV3-Large | 4.21M | 72.42% | 67.39% | 70.2 min |

**Selected model:** EfficientNet-B0 — highest accuracy and macro F1 among the three, with faster training than ResNet50.

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Scheduler | ReduceLROnPlateau (patience=5) |
| Batch size | 32 |
| Epochs | 30 |
| Mixed precision | Enabled |
| Dropout (head) | 0.2 |
| Loss | Weighted CrossEntropyLoss |
| Hardware | Google Colab T4 GPU |

Head design: `Dropout(0.2) → Linear(1280 → 7)`, replacing the ImageNet 1000-class output.

---

## Evaluation Metrics (Validation Set, Best Checkpoint)

| Metric | Value |
|---|---|
| Accuracy | 73.92% |
| Macro F1 | 69.81% |

**Per-class performance** (from 5-epoch diagnostic run, indicative of relative class difficulty):

| Class | Recall | Note |
|---|---|---|
| Surprise | 0.83 | Strongest — visually distinctive |
| Happy | 0.76 | Strong — largest class |
| Disgust | 0.74 | Strong despite smallest FER2013 class (weighted sampling effective) |
| Angry | 0.68 | Moderate |
| Neutral | 0.64 | Moderate |
| Fear | 0.55 | Weaker — visually ambiguous with Sad/Neutral |
| Sad | 0.44 | Weakest — confused with Fear (18%) and Angry (15%) |

---

## Known Limitations

1. **Residual class imbalance:** Weighted loss and sampling improve but do not eliminate imbalance effects — Sad and Fear remain the hardest classes.
2. **Domain gap:** FER2013's lab-controlled, low-resolution (48×48) images differ substantially from RAF-DB's in-the-wild, higher-resolution (100×100) images. Upscaling FER2013 to 224×224 introduces blur that may disadvantage FER2013-sourced predictions specifically.
3. **Colab session constraints:** Training was constrained by free-tier Colab GPU quotas and session disconnects. A resume-from-checkpoint mechanism (saving to Google Drive every epoch) was implemented to mitigate data loss from interruptions.
4. **Cross-domain generalization:** Performance on AffectNet (held-out, out-of-distribution test set) is evaluated separately by the team's evaluation stage — the in-distribution vs. out-of-distribution accuracy gap is the project's central research question and is not yet reflected in this model card.
5. **Demographic bias:** RAF-DB is demographically Western-skewed. Per-demographic accuracy analysis on RAF-DB is planned but not yet completed.

---

## Ethical Considerations

- No personally identifiable information is retained; all training images are from publicly available academic datasets (FER2013, RAF-DB) used under their respective research licenses.
- Facial emotion recognition models can encode and amplify demographic biases present in training data. Users should not treat model outputs as ground truth for an individual's emotional state, particularly across demographic groups underrepresented in FER2013 and RAF-DB.
- This model is not validated for clinical, legal, or safety-critical use.

---

## Reproducibility

```bash
git clone <repo-url>
cd mai204-face-analysis
pip install -r requirements.txt
dvc pull
dvc repro prepare
dvc repro train
```

Model checkpoint: `results/models/best_model.pth` (EfficientNet-B0)
Full ablation results: `results/ablation_table.json`
Training script: `src/train.py`
Dataset pipeline: `src/dataset.py`

---

## Citation

If referencing this work, cite the course project:

> Anand, A., Kaur, D., Zabalawi, A. R. (2026). *Real-Time Face Analysis: Emotion Recognition Under Varying Conditions.* MAI204 Computer Vision, Seneca Polytechnic.

Key literature referenced:
- Goodfellow et al. (2015) — FER2013
- Li, Deng & Du (2017) — RAF-DB
- Tan & Le (2019) — EfficientNet
- He et al. (2016) — ResNet
- Howard et al. (2019) — MobileNetV3
- Tutuianu et al. (2023) — class imbalance benchmarking methodology

---

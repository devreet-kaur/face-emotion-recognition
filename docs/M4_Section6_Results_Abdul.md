# M4 — Section 6: Results

**Author:** Abdul Raouf Zabalawi (Evaluation & Inference Lead)
**Status:** structure and analysis complete; numeric cells fill from
`results/affectnet_eval.json` and `results/robustness_results.json` once
`dvc repro evaluate` runs against the ablation checkpoint.

> Every `__` cell below maps to a specific JSON key, named in the caption. Fill them by
> running `dvc repro evaluate` — do not transcribe numbers by hand from console output.

---

## 6.1 Evaluation protocol

Three properties make the numbers in this section meaningful rather than decorative.

**AffectNet is held out absolutely.** It is used neither for training, nor for
validation, nor for hyperparameter or threshold selection. The model sees it exactly
once, in a single evaluation pass. Any protocol that tunes on the target domain — even
by selecting a decision threshold — converts a generalisation measurement into an
optimisation result, and we avoid that.

**The label space is reconciled, not coerced.** AffectNet ships eight categories; the
merged FER2013 + RAF-DB label set has seven. Contempt is *dropped* rather than folded
into a neighbouring class, because forcing it into e.g. Disgust would inject label noise
that is indistinguishable from model error in the final metric. The surviving 7-way
mapping is declared in `params.yaml → evaluate.affectnet_class_map` and is therefore
version-controlled and auditable.

**Everything is parameterised.** No degradation constant, class map, or path appears in
a `.py` file; all live in `params.yaml`. A reviewer can reproduce any cell in this
section by checking out the commit and running `dvc repro evaluate`.

---

## 6.2 Table 1 — Cross-domain performance: the lab-to-real gap

*Source: `results/affectnet_eval.json`*

| Evaluation set | Accuracy | Macro F1 | n |
|---|---|---|---|
| In-distribution test split (FER2013 + RAF-DB, 15%) | `__` | `__` | `__` |
| Held-out AffectNet validation subset | `__` | `__` | `__` |
| **Domain gap (in-distribution − AffectNet)** | **`__`** | **`__`** | — |

*Keys: `in_distribution.accuracy`, `in_distribution.f1`, `in_distribution.n_samples`;
`affectnet.accuracy`, `affectnet.f1`, `affectnet.n_samples`; `domain_gap_accuracy`,
`domain_gap_f1`.*

### Per-class breakdown on AffectNet

*Source: `affectnet.per_class` in the same file*

| Emotion | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Angry | `__` | `__` | `__` | `__` |
| Disgust | `__` | `__` | `__` | `__` |
| Fear | `__` | `__` | `__` | `__` |
| Happy | `__` | `__` | `__` | `__` |
| Neutral | `__` | `__` | `__` | `__` |
| Sad | `__` | `__` | `__` | `__` |
| Surprise | `__` | `__` | `__` | `__` |

**Figure 6.1** — `results/plots/confusion_matrix.png`. Row-normalised confusion matrix
on held-out AffectNet, with raw counts in parentheses. Read the off-diagonal mass: FER
confusions are typically not uniform, and neighbouring negative-valence classes
(Fear/Surprise, Sad/Neutral) exchange more probability mass than distant pairs.

### Interpretation

The domain gap is the headline result of this project. It is the difference between what
a benchmark number promises and what a deployed model delivers, and it is the quantity
almost never reported in the FER literature, where accuracy is overwhelmingly quoted on
a held-in test split of the same corpus used for training.

Two mechanisms contribute, and honesty requires separating them:

1. **Genuine domain shift.** AffectNet is scraped from the open web. Its images differ
   from FER2013 and RAF-DB in capture device, resolution, pose distribution, lighting,
   and demographic composition. A model that has only ever seen the training corpora has
   correspondingly narrow coverage.

2. **Label noise.** AffectNet's annotations are automatically harvested and are
   measurably noisier than RAF-DB's manual annotation. Some fraction of the drop is
   therefore a property of the *labels*, not of the model.

We do not claim the whole gap as evidence of model fragility. Section 6.3 exists
precisely to isolate the fragility component under conditions where the labels are ours
and are held constant.

---

## 6.3 Table 2 — Robustness under controlled degradation

*Source: `results/robustness_results.json`*

Applied to the in-distribution test split, so the labels are our own and unchanged. Any
accuracy movement is attributable to the corruption rather than to annotation quality —
which is what makes this table complementary to Table 1 rather than a repeat of it.

| Condition | Parameter | Accuracy | Macro F1 | Δ accuracy vs clean |
|---|---|---|---|---|
| Clean baseline | none | `__` | `__` | 0.0000 |
| Low light | brightness × 0.4 | `__` | `__` | `__` |
| Motion blur | Gaussian σ = 3.0 | `__` | `__` | `__` |
| Rotation | 30° | `__` | `__` | `__` |
| Occlusion | 40 × 40 px patch | `__` | `__` | `__` |

*Keys: `conditions.<name>.accuracy`, `.f1`, `.accuracy_drop_vs_clean`,
`.parameter`. Most damaging condition: `most_damaging_condition`.*

### Table 3 — Per-emotion F1 degradation

*Source: `worst_condition_per_emotion` and `conditions.<name>.per_class` in the same
file*

| Emotion | Clean F1 | Dark | Blur | Rotate | Occlude | Worst condition |
|---|---|---|---|---|---|---|
| Angry | `__` | `__` | `__` | `__` | `__` | `__` |
| Disgust | `__` | `__` | `__` | `__` | `__` | `__` |
| Fear | `__` | `__` | `__` | `__` | `__` | `__` |
| Happy | `__` | `__` | `__` | `__` | `__` | `__` |
| Neutral | `__` | `__` | `__` | `__` | `__` | `__` |
| Sad | `__` | `__` | `__` | `__` | `__` | `__` |
| Surprise | `__` | `__` | `__` | `__` | `__` | `__` |

Reported per class rather than as a single average, because averaging destroys the
finding. A 5-point mean drop is uninformative; a 5-point mean drop composed of a
20-point collapse in one class and near-immunity in the others is a statement about
*where* the model's evidence lives.

### Why these four parameter values

- **Low light (× 0.4).** Matches the strong-dark condition in Devreet's preprocessing
  benchmark, so the two studies are directly comparable: her benchmark asks which
  enhancement method best recovers a degraded image, this suite asks how much accuracy
  is lost without any enhancement. Same corruption, complementary questions.
- **Rotation (30°).** Training augments to ±15°. Testing inside the augmentation range
  would only confirm that the augmentation worked. 30° sits deliberately outside it,
  which is the only setting under which the result speaks to generalisation.
- **Blur (σ = 3.0).** The dominant real failure mode for a webcam deployment, and the
  condition the live demo hits most often in practice.
- **Occlusion (40 × 40 px).** Scaled to preserve a constant occluded *fraction* across
  input sizes, so the number is not an artefact of resolution.

### Cross-reference to the explainability study

Table 3 and Devreet's Pointing Game scores are two independent measurements of the same
underlying property: which facial region carries the evidence for each emotion. If
occlusion collapses Happy while barely moving Angry, the explainability heatmaps should
independently show Happy's evidence concentrated in the mouth region and Angry's around
the eyes and brow.

Agreement between the two is mutual corroboration from methodologically unrelated
directions — one perturbation-based, one gradient-based. **Disagreement is a bug
signal** in one of the two pipelines and should be investigated rather than reported.
This cross-check is a deliberate design property of the evaluation suite, not a
coincidence.

---

## 6.4 Deployment verification

The evaluation numbers describe a checkpoint; this subsection establishes that the
checkpoint is actually servable.

| Item | Verification | Status |
|---|---|---|
| FastAPI service | `/health` 200, `/classes` returns 7 labels, `/predict` returns label + confidence + all 7 scores | `__` |
| Test suite | `python -m pytest tests/ -v` → 9 passed | 9 passed |
| Lint | `python -m ruff check src/ tests/` | clean |
| Container | `docker build` then `/health` smoke test | `__` |
| CI | ruff → pytest → docker build → smoke test, green on `develop` | `__` |
| Drift monitor | `reports/drift/drift_report_*.html` generated | `__` |
| Live demo | `demo/webcam_demo.py` on webcam; 3-minute recording in Drive | `__` |

Two design decisions in this layer are worth recording, because both were failures found
by verification rather than by reading the code:

**Degraded start.** The service starts and reports `model_loaded: false` when the
checkpoint is absent, rather than raising on startup. The checkpoint is DVC-tracked and
so is legitimately absent in CI, in a fresh clone, and inside the image. Raising meant
the container never became healthy and the CI smoke test could not pass at any point
before the ablation finished.

**Validation ordering.** Upload validation precedes the model-availability check, so a
malformed request is always answered with 422 and never masked by a 503 about the model.
The original ordering caused four of the nine API tests to fail with 503 where they
expected 200 and 422.

**Drift monitoring closure.** The monitor originally consumed two CSVs that nothing in
the project produced, making the deliverable unreachable. `src/app.py` now logs one row
of image statistics per detected face, and `src/monitor.py --make-reference` derives the
training-distribution reference from the same deterministic split the model trained on.
The feature computation is duplicated in both files and asserted equivalent, because a
silent divergence there would render every drift number meaningless.

---

## 6.5 Limitations

1. **AffectNet label quality bounds the interpretation of Table 1.** The domain gap is
   an upper bound on model-attributable degradation, not a point estimate of it.
2. **Single evaluation pass, no confidence intervals.** Accuracy on a finite validation
   subset carries sampling error that is not quantified here. Bootstrap resampling over
   the AffectNet subset would give an interval and is the obvious extension.
3. **Degradations are synthetic.** Gaussian blur is not motion blur; a black rectangle
   is not a hand. The suite establishes ordinal sensitivity, not calibrated real-world
   failure rates.
4. **One corruption at a time.** Real deployment conditions compound — dim *and* blurred
   *and* off-axis simultaneously. Compositional degradation is untested.
5. **Colab session limits.** Training duration was bounded by the platform, so the
   checkpoint under evaluation is not trained to convergence. The gap reported here is
   a property of this checkpoint, not of the architecture in the limit.
6. **Contempt is out of scope.** Dropping it means the system cannot be compared
   directly against 8-class AffectNet results in the literature.

---

## 6.6 Conclusions

*Complete after the numbers land. The three claims this section must support:*

1. **The gap is the finding.** State the in-distribution accuracy, the AffectNet
   accuracy, and the difference — and say plainly that the second number is the one that
   describes deployment.
2. **Fragility is class-specific and localised.** Name the most damaging condition and
   the emotion it hurts most, and connect that to the explainability evidence for the
   same region.
3. **The pipeline is reproducible and servable.** A fresh clone, `dvc pull`,
   `dvc repro`, nine passing tests, and a container answering `/health` — verified end
   to end, not asserted.

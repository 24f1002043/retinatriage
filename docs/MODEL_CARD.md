# Model Card — RetinaTriage DR Classifier

> **This model is not a medical device.** It is a student research and
> engineering project. It has never been validated on a clinical population,
> has no regulatory clearance, and must not be used to make or support any
> real diagnostic or treatment decision.

Model cards follow the format proposed in
[Mitchell et al., *Model Cards for Model Reporting* (2019)](https://arxiv.org/abs/1810.03993).

---

## 1. Model details

| Field | Value |
| --- | --- |
| Task | Ordinal 5-class diabetic retinopathy grading from colour fundus photographs |
| Architecture | ResNet50, ImageNet-pretrained, final FC layer replaced with a 5-way classifier |
| Transfer learning | All layers fine-tuned (no frozen backbone) |
| Input | RGB fundus image, resized to 224 × 224, normalised with ImageNet statistics |
| Output | Softmax distribution over 5 grades + Grad-CAM saliency overlay |
| Loss | Cross-entropy weighted by inverse class frequency |
| Optimiser | Adam, cosine-annealed learning rate |
| Model selection | Best epoch by **validation quadratic weighted kappa** |
| Framework | PyTorch / torchvision |
| Licence | MIT |

### Grades

| Grade | Label | Clinical meaning |
| --- | --- | --- |
| 0 | No DR | No visible diabetic retinopathy |
| 1 | Mild | Mild nonproliferative DR (microaneurysms only) |
| 2 | Moderate | Moderate nonproliferative DR |
| 3 | Severe | Severe nonproliferative DR |
| 4 | Proliferative DR | Neovascularisation present |

---

## 2. Intended use

**Intended:** demonstrating an end-to-end explainable medical-imaging pipeline —
training, evaluation on an ordinal metric, inference serving, and saliency-based
explanation. Educational and portfolio use.

**Out of scope, explicitly:**

- Any clinical, diagnostic, screening, or triage decision about a real patient.
- Use on image types the model never saw: OCT, fluorescein angiography,
  slit-lamp photography, smartphone fundus captures from unvalidated adapters.
- Grading eyes with comorbid pathology (glaucoma, AMD, cataract) — the model has
  no concept of these and will still emit a confident DR grade.

---

## 3. Training data

[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
— 3,662 labelled fundus photographs collected by Aravind Eye Hospital in rural
India, graded 0–4 by trained clinicians.

**Split:** 85 % train / 15 % validation, stratified by grade, seeded at 42 so
the split is reproducible by `evaluate.py`.

**Augmentation (train only):** horizontal and vertical flips, ±20° rotation,
brightness/contrast/saturation jitter at 0.2.

### Known data limitations

- **Severe class imbalance.** Grade 0 is roughly half the dataset; grades 3 and 4
  are a few hundred images each. This is why the loss is frequency-weighted, and
  why accuracy is not the reported metric.
- **Single-source data.** One hospital network, one geography, largely one
  camera pipeline. Performance on images from other populations, ethnicities, or
  capture devices is **unmeasured and should be assumed worse.**
- **Single-grader labels.** Inter-grader disagreement on DR staging is
  substantial in the literature; the ceiling on this task is not 1.0 kappa.
- **No patient-level grouping.** The split is per-image. If the dataset contains
  both eyes of the same patient, some leakage across the split is possible.

---

## 4. Evaluation

**Primary metric: quadratic weighted kappa (QWK).** DR grades are *ordinal* —
grading a stage-4 eye as stage 3 is a much smaller error than grading it stage 0.
Accuracy treats those two mistakes identically; QWK penalises by the square of
the grade distance. It is also the metric the original Kaggle competition scored.

**Secondary:** per-class precision / recall / F1, the confusion matrix, and
referable-DR sensitivity and specificity (grades ≥ 2 collapsed against < 2),
which is the view closest to how a screening tool would actually be judged.

### Reproducing the numbers

```bash
cd training
python evaluate.py --data_dir data/ --weights ../backend/weights/model.pth --out_dir reports/
```

This writes `reports/eval_metrics.json` and `reports/confusion_matrix.png`,
rebuilding the exact validation split from the seed.

### Results

<!-- Replace this block with the real output of evaluate.py after training. -->

| Metric | Value |
| --- | --- |
| Quadratic weighted kappa | _run `evaluate.py` to fill in_ |
| Accuracy | _run `evaluate.py` to fill in_ |
| Referable-DR sensitivity (grade ≥ 2) | _run `evaluate.py` to fill in_ |
| Referable-DR specificity | _run `evaluate.py` to fill in_ |

Typical QWK for this architecture and dataset lands in the **0.75 – 0.85** range;
treat that as an expectation to verify, not a claim about this checkpoint.

---

## 5. Explainability

Every prediction is returned with a **Grad-CAM** overlay
([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)) computed on the
final bottleneck block of `layer4`, targeting the predicted class.

**How to read it — and how not to.** Grad-CAM shows which regions most increased
the score for the predicted grade. It is a coarse 7 × 7 map upsampled to image
resolution, so it localises *approximately*. Critically, a plausible-looking
heatmap is **not** evidence the prediction is correct: a model can attend to the
right region and still grade it wrong, or attend to an artifact and be right by
coincidence. The overlay is a debugging and communication aid, not a
justification.

---

## 6. Ethical considerations and risks

- **Automation bias.** A confident-looking percentage next to a medical image
  invites over-trust. The UI carries a persistent disclaimer banner, and flags
  every response when no trained weights are loaded, precisely because a demo
  screenshot can otherwise be mistaken for a real result.
- **False negatives are the dangerous error.** Missing proliferative DR is
  sight-threatening; a false positive costs an unnecessary referral. Any
  deployment-shaped version of this would need a decision threshold tuned for
  sensitivity, not the argmax used here.
- **Fairness is unmeasured.** The dataset carries no demographic metadata, so
  no subgroup performance analysis was possible. Absence of measured bias is not
  evidence of absence of bias.
- **Confidence is not calibrated.** Softmax outputs are reported raw. No
  temperature scaling or calibration analysis was performed, so a "87 %" should
  not be read as an 87 % probability of being correct.

---

## 7. Caveats

- The checkpoint in `backend/weights/` is not distributed with this repository;
  train your own with `training/train.py`.
- Without weights the API deliberately still runs, on the ImageNet backbone, and
  sets `untrained_warning: true` on every response so pipeline testing can never
  be confused with real inference.
- Grad-CAM requires a backward pass, so inference latency is roughly double what
  a bare forward pass would cost.

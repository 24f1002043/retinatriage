# Training the RetinaTriage Model

Train a ResNet50-based diabetic retinopathy classifier on the
[APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
dataset (5 classes: No DR → Proliferative DR).

---

## Prerequisites

- **Python 3.9+**
- **NVIDIA GPU recommended** — training on CPU is very slow
- **Kaggle account** — required to download the competition dataset

---

## Step 1: Install Dependencies

```bash
cd training
pip install -r requirements.txt
```

---

## Step 2: Get the APTOS 2019 Dataset from Kaggle

### 2a. Create a Kaggle API token

1. Go to [kaggle.com](https://www.kaggle.com/) → click your profile icon → **Settings**
2. Scroll to the **API** section → click **Create New API Token**
3. This downloads a `kaggle.json` file — keep it safe

### 2a-alt. Or skip the competition entirely

Joining the competition is only needed for the official download. A community
re-upload of the same data works without it and is a single command:

```bash
kaggle datasets download -d mariaherrerot/aptos2019
unzip aptos2019.zip -d data/
```

Mirrors rename things — `train_1.csv` instead of `train.csv`, `train/` instead
of `train_images/`, JPEG instead of PNG, `image`/`level` column headers instead
of `id_code`/`diagnosis`. `train.py` and `evaluate.py` detect all of those
automatically. If a layout still isn't recognised, the error lists what is
actually in the directory so you can point at it directly:

```bash
python train.py --data_dir data/ --csv data/train_1.csv --images_dir data/train --out ../backend/weights/model.pth
```

Check what you got before training:

```bash
python -c "import pathlib; print([p.name for p in pathlib.Path('data').iterdir()])"
```

> Whatever layout you train on, pass the **same** `--csv`, `--images_dir`,
> `--seed`, and `--val_size` to `evaluate.py`, or it will rebuild a different
> split and report on data the model has already seen.

### 2b. Accept the competition rules

> **Important:** You **must** visit
> <https://www.kaggle.com/competitions/aptos2019-blindness-detection>
> and click **Join Competition** / accept the rules **before** downloading.
> Otherwise the download will fail with a 403 error.

### 2c. Download & extract

```bash
pip install kaggle

# Linux / macOS
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
kaggle competitions download -c aptos2019-blindness-detection

# Extract into data/
unzip aptos2019-blindness-detection.zip -d data/
```

After extraction your directory should look like:

```
training/
  data/
    train.csv            # columns: id_code, diagnosis (0-4)
    train_images/        # {id_code}.png files
    test.csv
    test_images/
  train.py
  requirements.txt
```

---

## Step 3: Train the Model

### Full training run

```bash
python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out ../backend/weights/model.pth
```

### Smoke test (verify the pipeline works end-to-end)

```bash
python train.py --data_dir data/ --smoke_test --out ../backend/weights/model.pth
```

This uses only 100 images for 1 epoch — takes under a minute on any machine.

### All CLI arguments

| Argument        | Default              | Description                                          |
| --------------- | -------------------- | ---------------------------------------------------- |
| `--data_dir`    | *(required)*         | Path to the `data/` directory                        |
| `--csv`         | auto-detect          | Labels CSV, if the layout isn't recognised           |
| `--images_dir`  | auto-detect          | Image folder, if the layout isn't recognised         |
| `--epochs`      | `15`                 | Number of training epochs                            |
| `--batch_size`  | `32`                 | Batch size                                           |
| `--lr`          | `1e-4`               | Learning rate (Adam)                                 |
| `--out`         | `weights/model.pth`  | Where to save the best model weights                 |
| `--val_size`    | `0.15`               | Fraction held out for validation                     |
| `--num_workers` | `2`                  | DataLoader workers — **use `0` on Windows** if it hangs |
| `--seed`        | `42`                 | Random seed; controls the split and all RNGs         |
| `--smoke_test`  | off                  | Quick sanity run: 100 images, 1 epoch                |

### What training produces

Alongside the checkpoint, each run writes a `metrics.json` next to `--out`
containing the per-epoch history, the confusion matrix, and the per-class
report for the best epoch.

The checkpoint is a dict — `state_dict` plus metadata (architecture, class
names, image size, normalisation statistics, best epoch, kappa, seed) — so a
weights file is self-describing rather than an anonymous blob. It contains only
tensors and plain primitives, which lets the backend load it with
`torch.load(..., weights_only=True)` instead of unpickling arbitrary objects.

---

## Step 4: Google Colab (Free GPU)

If you don't have a local GPU, use Google Colab's free T4.
Open a new Colab notebook and paste:

```python
# Step 1: Upload your kaggle.json
from google.colab import files
uploaded = files.upload()  # upload kaggle.json

# Step 2: Setup Kaggle
!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

# Step 3: Download dataset
!pip install kaggle
!kaggle competitions download -c aptos2019-blindness-detection
!unzip -q aptos2019-blindness-detection.zip -d data/

# Step 4: Clone/upload training code
# Upload train.py and requirements.txt, or clone your repo
!pip install -r requirements.txt

# Step 5: Train
!python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out model.pth

# Step 6: Download the trained model
files.download('model.pth')
```

---

## Step 5: Evaluate the Model

`evaluate.py` rebuilds the **exact** validation split `train.py` used — same
seed, same stratification — so the reported numbers are on genuinely held-out
data:

```bash
python evaluate.py --data_dir data/ --weights ../backend/weights/model.pth --out_dir reports/
```

It reports:

| Metric | Why it's here |
| ------ | ------------- |
| **Quadratic weighted kappa** | Primary metric. DR grades are ordinal, so errors are penalised by the square of the grade distance — and it's what the Kaggle competition scored. |
| Accuracy | Reported for context only; it is misleading on data this imbalanced. |
| Per-class precision / recall / F1 | Shows whether the rare severe grades are actually detected, or just absorbed into their neighbours. |
| Confusion matrix | Where the errors land. Saved as JSON and as a PNG. |
| Referable-DR sensitivity & specificity | Grades ≥ 2 collapsed against < 2 — the screening-relevant view a clinician would ask about. |

Outputs land in `reports/`: `eval_metrics.json` and `confusion_matrix.png`.

> Pass the same `--seed` and `--val_size` you trained with. Different values
> rebuild a different split, and the model will have seen some of it.

---

## Step 6: Deploy the Model

Copy the trained checkpoint into the backend so the web app picks it up on
startup:

```bash
cp model.pth ../backend/weights/model.pth
```

The backend loads it in `backend/model.py`, accepting either a bare
`state_dict` or the metadata dict this script writes:

```python
checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict)
```

Restart the backend and check that the weights were found:

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"untrained":false,"device":"cpu"}
```

If `untrained` is still `true`, the file isn't where the backend expects it —
set `RETINATRIAGE_WEIGHTS` to point at it explicitly.

---

## Expected Results

| Metric                          | Expected Range   |
| ------------------------------- | ---------------- |
| Validation quadratic weighted κ | **0.75 – 0.85**  |
| Validation accuracy             | **70 – 80 %**    |
| Training time (Colab T4)        | ~30 – 45 minutes |

> **Note:** Results vary with random seed and data shuffling.
> The quadratic weighted kappa (QWK) is the primary metric — it accounts
> for the ordinal nature of DR severity grades and is what the original
> Kaggle competition used for scoring.

# Training

Trains a ResNet50 diabetic retinopathy classifier on the [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection) dataset (5 classes, No DR to Proliferative DR).

## Requirements

- Python 3.9+
- NVIDIA GPU recommended, CPU training is very slow
- Kaggle account to download the data

## 1. Install

```bash
cd training
pip install -r requirements.txt
```

## 2. Get the data

### Kaggle API token

1. On kaggle.com go to your profile, then Settings.
2. In the API section click Create New API Token.
3. This downloads `kaggle.json`.

### Option: use a mirror instead of the competition

Joining the competition is only needed for the official download. A community re-upload works without it:

```bash
kaggle datasets download -d mariaherrerot/aptos2019
unzip aptos2019.zip -d data/
```

Mirrors rename things: `train_1.csv` instead of `train.csv`, `train/` instead of `train_images/`, JPEG instead of PNG, `image`/`level` columns instead of `id_code`/`diagnosis`. `train.py` and `evaluate.py` detect these. If the layout still isn't recognised, the error lists what's in the folder and you can pass the paths directly:

```bash
python train.py --data_dir data/ --csv data/train_1.csv --images_dir data/train --out ../backend/weights/model.pth
```

To check what you downloaded:

```bash
python -c "import pathlib; print([p.name for p in pathlib.Path('data').iterdir()])"
```

Whatever layout you use, pass the same `--csv`, `--images_dir`, `--seed` and `--val_size` to `evaluate.py`. Otherwise it builds a different split and evaluates on data the model has seen.

### Accept the competition rules

For the official download you have to open https://www.kaggle.com/competitions/aptos2019-blindness-detection and join the competition first, otherwise you get a 403.

### Download and extract

```bash
pip install kaggle

# Linux / macOS
mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

kaggle competitions download -c aptos2019-blindness-detection
unzip aptos2019-blindness-detection.zip -d data/
```

Expected layout:

```
training/
  data/
    train.csv            # columns: id_code, diagnosis (0-4)
    train_images/        # {id_code}.png
    test.csv
    test_images/
  train.py
  requirements.txt
```

## 3. Train

Full run:

```bash
python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out ../backend/weights/model.pth
```

Smoke test (100 images, 1 epoch, under a minute):

```bash
python train.py --data_dir data/ --smoke_test --out ../backend/weights/model.pth
```

### Arguments

| Argument        | Default              | Description                                    |
| --------------- | -------------------- | ---------------------------------------------- |
| `--data_dir`    | required             | path to `data/`                                |
| `--csv`         | auto-detect          | labels CSV if the layout isn't recognised      |
| `--images_dir`  | auto-detect          | image folder if the layout isn't recognised    |
| `--epochs`      | `15`                 | number of epochs                               |
| `--batch_size`  | `32`                 | batch size                                     |
| `--lr`          | `1e-4`               | learning rate (Adam)                           |
| `--out`         | `weights/model.pth`  | where to save the best weights                 |
| `--val_size`    | `0.15`               | validation fraction                            |
| `--num_workers` | `2`                  | DataLoader workers, use `0` on Windows if it hangs |
| `--seed`        | `42`                 | seed for the split and all RNGs                |
| `--smoke_test`  | off                  | 100 images, 1 epoch                            |

### Output

Each run writes a `metrics.json` next to `--out` with the per-epoch history, confusion matrix and per-class report for the best epoch.

The checkpoint is a dict with `state_dict` and metadata (architecture, class names, image size, normalisation stats, best epoch, kappa, seed). It only contains tensors and plain values, so the backend can load it with `torch.load(..., weights_only=True)`.

## 4. Google Colab

Without a local GPU you can use Colab's free T4. In a new notebook:

```python
from google.colab import files
uploaded = files.upload()  # upload kaggle.json

!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

!pip install kaggle
!kaggle competitions download -c aptos2019-blindness-detection
!unzip -q aptos2019-blindness-detection.zip -d data/

# upload train.py and requirements.txt, or clone the repo
!pip install -r requirements.txt

!python train.py --data_dir data/ --epochs 15 --batch_size 32 --lr 1e-4 --out model.pth

files.download('model.pth')
```

## 5. Evaluate

`evaluate.py` rebuilds the same validation split as `train.py` (same seed and stratification), so the numbers are on held-out data:

```bash
python evaluate.py --data_dir data/ --weights ../backend/weights/model.pth --out_dir reports/
```

| Metric | Why |
| ------ | --- |
| Quadratic weighted kappa | main metric. Grades are ordinal so errors are penalised by squared grade distance. Also what the Kaggle competition used. |
| Accuracy | for reference only, misleading on imbalanced data |
| Per-class precision / recall / F1 | shows whether rare severe grades are detected |
| Confusion matrix | where errors happen, saved as JSON and PNG |
| Referable-DR sensitivity / specificity | grade >= 2 vs < 2, the screening view |

Outputs go to `reports/`: `eval_metrics.json` and `confusion_matrix.png`.

Use the same `--seed` and `--val_size` as in training, otherwise the split changes and includes training data.

## 6. Use the model in the app

Copy the checkpoint into the backend:

```bash
cp model.pth ../backend/weights/model.pth
```

`backend/model.py` loads it and accepts either a plain `state_dict` or the dict this script saves:

```python
checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict)
```

Restart the backend and check:

```bash
curl http://localhost:8000/health
# {"status":"ok","model_loaded":true,"untrained":false,"device":"cpu"}
```

If `untrained` is still `true`, the file isn't where the backend expects it. Set `RETINATRIAGE_WEIGHTS` to its path.

## Results

| Metric                          | Expected range   | Got (seed 42) |
| ------------------------------- | ---------------- | ------------- |
| Validation quadratic weighted κ | 0.75 - 0.85      | 0.9361        |
| Validation accuracy             | 70 - 80 %        | 84.8 %        |
| Training time (Colab T4)        | ~30 - 45 min     | ~60 min       |

This is a 15-epoch run on the training part of the `mariaherrerot/aptos2019` mirror, validated on a seeded 15 % split. That split is easier than the competition test set, so it's an in-domain validation number and not comparable to the leaderboard. The per-class results in the [Model Card](../docs/MODEL_CARD.md) are weaker than the overall kappa.

Results change with the seed and shuffling.

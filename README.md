# casting-defect-MLOps
This project is about building the full lifecycle infrastructure around a computer vision model, not just training one. The goal is to simulate what a production ML team does: data versioning,
reproducible training, automated evaluation gates, CI/CD, containerized serving, and
drift monitoring.

Built sequentially through seven stages, documented below as each one completes.

## Task & data

- **Task**: binary image classification — `def_front` (defective) vs `ok_front` (ok) —
  on the [Kaggle casting product image dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (~7,000 images, 300×300 RGB).
- **Planned model**: ResNet18 (transfer learning, PyTorch).

## Stack

| Concern | Tool |
|---|---|
| Data versioning | DVC (local remote for now) |
| Experiment tracking | MLflow |
| Training | PyTorch, ResNet18 transfer learning |
| CI/CD | GitHub Actions |
| Serving | FastAPI |
| Containerization | Docker, GHCR |
| Monitoring | Prometheus, Grafana |

## Stage plan

| Stage | Scope | Status |
|---|---|---|
| 1 | Repo scaffolding, EDA, leakage check | ✅ Done |
| 2 | Data versioning with DVC, simulated time-batched data arrival | ✅ Done |
| 3 | PyTorch training pipeline (ResNet18) + MLflow tracking | ⏳ Not started |
| 4 | Evaluation gate vs. stored baseline | ⏳ Not started |
| 5 | GitHub Actions CI/CD | ⏳ Not started |
| 6 | FastAPI serving | ⏳ Not started |
| 7 | Prediction logging + drift detection dashboard | ⏳ Not started |

---

## Stage 1 — EDA & data integrity

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

**1. Class balance**
- Train: `def_front`=3758, `ok_front`=2875 (~57/43 split)
- Test (raw, pre-cleaning): `def_front`=453, `ok_front`=262 (~63/37 split)
- Moderate imbalance → decided to use per-class precision/recall/F1 (or balanced
  accuracy) as the primary metric rather than raw accuracy, and to consider
  class-weighted loss in Stage 3.

**2. Image dimensions & pixel statistics**
- All images uniform: 300×300 RGB, no corrupt or off-size files.
- Train/test are consistent per class (e.g. `def_front` mean pixel ≈139.3 train vs
  ≈138.9 test) — no distribution shift between splits on this measure.
- `def_front` skews darker than `ok_front` (≈139 vs ≈150 mean pixel value), with
  moderate overlap — physically sensible, since defects appear as dark regions.
- Neither mean nor std of pixel intensity cleanly separates the two classes on its
  own, ruling out a simple brightness/contrast threshold and supporting the choice
  of a CNN that can learn the spatial location of a defect (Stage 3).

**3. Sample image grids** — visual spot-check per class, saved to
`reports/figures/sample_grid.png`.

**4. Duplicate / leakage detection** — [`scripts/find_leakage.py`](scripts/find_leakage.py)
- Perceptual hashing (phash) was unreliable here (false positives/negatives on this
  dataset), so leakage detection used exact content hashing (MD5) instead.
- Found **64 exact-duplicate `ok_front` images** shared between train and raw test —
  i.e. true train/test leakage, not just visual similarity.

**Cleanup** — [`scripts/clean_splits.py`](scripts/clean_splits.py)
- Removed the 64 leaked images from test (train left unchanged).
- Leakage-free data written to `data/processed/` — **this is the source of truth**
  for all downstream stages (DVC tracking, training, evaluation):
  - `train`: unchanged (6633 images)
  - `test`: 651 images (453 `def_front`, 198 `ok_front`)
- Full audit trail of what was removed and why: `reports/leaked_files.txt`
  (md5 → removed test file → matching train file).

---

## Stage 2 — Data versioning with DVC

- DVC initialized on the repo; local remote configured at `~/dvc-remote/casting`
  (`.dvc/config`, `type = copy`).
- `data/processed/test` tracked as a single DVC output (`test.dvc`) — the frozen,
  leakage-free evaluation set (651 images, 6.6 MB).
- **Simulated time-batched training data**: rather than versioning `train` as one
  blob, [`scripts/make_batches.py`](scripts/make_batches.py) splits it into three
  stratified batches to simulate incremental data arrival over time:
  - Per-class shuffle (fixed seed, `SEED=42`) then sliced at 50% / 75% into
    `batch_1` / `batch_2` / `batch_3`, keeping each batch class-stratified.
  - `batch_1`: 3316 images (32.5 MB) — the initial training set
  - `batch_2`: 1658 images (16.3 MB) — first incremental arrival
  - `batch_3`: 1659 images (16.3 MB) — second incremental arrival
  - Each batch tracked as its own DVC output (`batch_1.dvc`, `batch_2.dvc`,
    `batch_3.dvc`), so later stages can check out and train against exactly the
    data available "as of" a given point in time.
- `data/raw/` stays untracked/local (excluded via `.gitignore`); `data/processed/`
  is DVC-tracked (raw `.jpg`/etc. ignored by Git, `.dvc` pointer files committed).

---

## Repo layout

```
data/
  raw/          # original Kaggle download (local only, not versioned)
  processed/    # leakage-free, DVC-tracked: train, test, batch_1/2/3
notebooks/
  01_eda.ipynb  # Stage 1 EDA
scripts/
  find_leakage.py   # content-hash based train/test & intra-split duplicate detection
  clean_splits.py   # builds data/processed/ with leaked test images removed
  make_batches.py   # splits train into 3 stratified time-batches for DVC
reports/
  figures/      # EDA plots (class balance, pixel intensity, sample grid)
  leaked_files.txt  # audit trail of removed leaked images
src/            # (empty so far — training/eval/serving code lands here from Stage 3)
tests/          # (empty so far)
models/         # (empty so far — trained model artifacts land here from Stage 3)
```

---

*This README is updated as each stage completes — see the status table above for
current progress.*

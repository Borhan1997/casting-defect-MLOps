# casting-defect-MLOps

Most of the work in a production ML system isn't the model. This repo builds the machinery around one: versioned data, reproducible training, an evaluation gate that can block a bad model, CI that runs all of it on every change, and (still to come) a containerized API with drift monitoring.

The task itself is simple: spot defective metal castings in photos. The pipeline around it is the interesting part.

The project is built in seven stages, and each one gets written up here when it's finished. Stages 1 to 5 are done.

## Task and data

Binary image classification, `def_front` (defective) against `ok_front` (fine), on the [Kaggle casting product image dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product). It has about 7,000 images, all 300x300 RGB.

The model is a ResNet18 pretrained on ImageNet, with only a new final layer trained (transfer learning, PyTorch).

## Stack

| Concern | Tool |
|---|---|
| Data versioning | DVC, with a DagsHub remote for CI and a local remote for development |
| Experiment tracking | MLflow (local SQLite backend) |
| Training | PyTorch, ResNet18 |
| CI/CD | GitHub Actions |
| Serving | FastAPI (planned) |
| Containerization | Docker, GHCR (planned) |
| Monitoring | Prometheus, Grafana (planned) |

## Stage plan

| Stage | Scope | Status |
|---|---|---|
| 1 | Repo scaffolding, EDA, leakage check | Done |
| 2 | Data versioning with DVC, simulated time-batched data arrival | Done |
| 3 | PyTorch training pipeline (ResNet18) and MLflow tracking | Done |
| 4 | Evaluation gate against a stored baseline | Done |
| 5 | GitHub Actions CI/CD | Done |
| 6 | FastAPI serving | Not started |
| 7 | Prediction logging and drift detection dashboard | Not started |

## Running it

Python 3.12 is what CI uses. The data lives in DVC, so a fresh clone has no images until you pull them.

```bash
pip install -r requirements.txt -r requirements-dev.txt

# Data. The DagsHub remote needs a token; CI sets it up with
# `dvc remote modify --local dagshub ...` (see .github/workflows/ci.yml).
dvc pull -r dagshub

# Training reads the validation images from data/processed/train, which is not
# tracked directly, so rebuild it from the three batches.
mkdir -p data/processed/train
cp -r data/processed/batch_1/. data/processed/train/
cp -r data/processed/batch_2/. data/processed/train/
cp -r data/processed/batch_3/. data/processed/train/

# Train on all three batches, then evaluate once on the test set, then gate.
python -m src.train --version v3
python -m src.evaluate --run-id <TRAINING_RUN_ID>
python gate/gate.py --eval-run-id <EVAL_RUN_ID> --tracking-uri sqlite:///mlflow.db

# Unit tests and lint
pytest
ruff check .
```

Run IDs show up in the MLflow UI (`mlflow ui --backend-store-uri sqlite:///mlflow.db`). Run everything from the repo root.

---

## Stage 1: EDA and data integrity

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

**Class balance**
- Train: `def_front`=3758, `ok_front`=2875 (about 57/43).
- Test, raw and before cleaning: `def_front`=453, `ok_front`=262 (about 63/37).
- The imbalance is moderate, so raw accuracy isn't a good headline number. Per-class precision, recall and F1 (or balanced accuracy) are the primary metrics instead, and class-weighted loss was on the table for Stage 3.

**Image dimensions and pixel statistics**
- Every image is 300x300 RGB. No corrupt or off-size files.
- Train and test agree per class. For example, `def_front` has a mean pixel value of about 139.3 in train and 138.9 in test, so there's no distribution shift between the splits on this measure.
- `def_front` images are darker than `ok_front` ones (about 139 against 150 mean pixel value), with a lot of overlap. That makes physical sense, since defects show up as dark regions.
- Neither the mean nor the standard deviation of intensity separates the classes cleanly. A brightness or contrast threshold won't work, which is the case for a CNN that can learn where a defect sits.

**Sample grids.** A visual spot-check per class, saved to `reports/figures/sample_grid.png`.

**Duplicates and leakage.** [`scripts/find_leakage.py`](scripts/find_leakage.py)
- Perceptual hashing (phash) gave both false positives and false negatives on this dataset, so the check uses exact content hashes (MD5).
- It found **64 exact-duplicate `ok_front` images** shared between train and the raw test set. That's real train/test leakage, not just visual similarity.

**Cleanup.** [`scripts/clean_splits.py`](scripts/clean_splits.py)
- The 64 leaked images were removed from test. Train is untouched.
- The cleaned data is written to `data/processed/`, which is the source of truth for everything downstream (DVC, training, evaluation):
  - `train`: unchanged, 6633 images
  - `test`: 651 images (453 `def_front`, 198 `ok_front`)
- `reports/leaked_files.txt` is the audit trail: md5, removed test file, matching train file.

---

## Stage 2: Data versioning with DVC

- DVC is initialized on the repo. The original remote is local, at `~/dvc-remote/casting` (`.dvc/config`, `type = copy`). A DagsHub remote was added in Stage 5 so CI can pull the data.
- `data/processed/test` is tracked as a single DVC output (`test.dvc`). It's the frozen, leakage-free evaluation set: 651 images, 6.6 MB.
- To simulate data arriving over time, [`scripts/make_batches.py`](scripts/make_batches.py) splits train into three batches instead of versioning it as one blob. Each class is shuffled with a fixed seed (`SEED=42`), then cut at 50% and 75%, so every batch keeps the class ratio:
  - `batch_1`: 3316 images (32.5 MB), the initial training set
  - `batch_2`: 1658 images (16.3 MB), the first top-up
  - `batch_3`: 1659 images (16.3 MB), the second top-up
- Each batch is its own DVC output (`batch_1.dvc`, `batch_2.dvc`, `batch_3.dvc`), so a later stage can train on exactly the data that existed "as of" a given point.
- `data/raw/` stays local and out of Git. In `data/processed/`, the images are ignored by Git and only the `.dvc` pointer files are committed.

---

## Stage 3: Training and experiment tracking

Code: [`src/train.py`](src/train.py), [`src/model.py`](src/model.py), [`src/dataset.py`](src/dataset.py), [`src/config.py`](src/config.py)

**Model.** ResNet18 with ImageNet weights. The whole backbone is frozen and the last layer is replaced by a single linear unit, so only 513 parameters train. The loss is `BCEWithLogitsLoss`, the optimizer is AdamW (lr 1e-3), and a run is 5 epochs at batch size 32 with seed 42. Images go through the standard ResNet18 preprocessing from torchvision.

**Labels.** `1` is `def_front` (defective) and `0` is `ok_front`. The dataset returns integer labels, and the conversion to float for the loss happens in the training loop.

**Validation.** A fixed validation set of 995 images (15% of train, stratified, seed 42) is carved out once by [`scripts/make_val_split.py`](scripts/make_val_split.py) and stored as `data/val_manifest.txt`. The manifest is a plain list of paths, so it lives in Git rather than DVC. The script also hashes the images to confirm none of them appear in the test set. Training uses a version's batches minus the manifest, so validation never leaks into training. `test/` isn't touched during training.

**Data versions.** `--version v1|v2|v3` picks the cumulative batches: v1 is `batch_1`, v2 adds `batch_2`, v3 adds `batch_3`. At v3 that's 5638 training images. All hyperparameters and paths sit in one frozen `TrainConfig` dataclass, and a version is just a modified copy of it.

**MLflow.** Each run logs its parameters (including `data_version` and the batch list as a comma-separated string), per-epoch train and validation metrics, the final metrics, and the model itself. The current git commit is attached as a `git_commit` tag, so any run can be traced back to the code that produced it. The tracking store is a local SQLite file, `mlflow.db`, which is not committed.

**BatchNorm experiment.** `--bn-eval` keeps the BatchNorm layers in eval mode during training. It exists as a flag so the effect can be compared run against run.

**Test evaluation.** [`src/evaluate.py`](src/evaluate.py) takes a training run ID, scores that model on the 651-image test set, and logs the result as a separate MLflow run tagged with the training run's ID. It asserts the exact test composition (651 images, 453 defective, 198 OK) before doing anything, and it refuses to evaluate the same model twice unless you pass `--force`. Looking at the test set repeatedly and keeping the best number would defeat the point of having one.

Notebooks [`02_dataset_sanity_check`](notebooks/02_dataset_sanity_check.ipynb) and [`03_model_sanity_check`](notebooks/03_model_sanity_check.ipynb) hold the checks done along the way.

---

## Stage 4: Evaluation gate

Code: [`gate/gate.py`](gate/gate.py), [`gate/common.py`](gate/common.py), [`gate/promote.py`](gate/promote.py), policy and baseline in [`gate/baseline.json`](gate/baseline.json)

The gate decides whether a candidate model is good enough to move forward, by comparing its test-set evaluation against a stored baseline.

**What it judges.** A missed defect is the expensive mistake in this problem, so the primary metric is defect recall. OK recall guards against a model that catches defects by flagging everything. Accuracy and test loss are logged and shown in the report, but they don't decide anything.

**The rule.** Recall is turned back into an error count (the gate checks that the count is consistent with the test set size), and a check passes when

```
candidate errors <= min(absolute floor, baseline errors + regression margin)
```

So a candidate has to clear a fixed minimum standard and also not get meaningfully worse than the current baseline. With the current numbers:

| Check | Baseline | Margin | Floor | Allowed |
|---|---|---|---|---|
| Missed defects | 11 | +3 | 20 | 14 (margin decides) |
| False alarms | 0 | +3 | 9 | 3 (margin decides) |

The floor only starts to matter if the baseline drifts far enough that baseline plus margin would exceed it.

**Baseline.** The v3 model is the baseline: 98.3% accuracy on test, defect recall 0.976 (11 of 453 defects missed), OK recall 1.0 (no false alarms in 198), test loss 0.0786. `baseline.json` also records the MLflow run IDs, the data version and the git commit, and the gate refuses to run if any of those are empty.

**Exit codes.** `0` pass, `1` fail (the model is too weak), `2` error (the evidence is missing or inconsistent, so nothing was judged). Keeping failure and error apart means a broken pipeline never looks like a bad model, or the reverse.

**Each candidate is scored once on `test/`.** Candidates are chosen and tuned on the validation set, and the gate is the last check, not a tuning loop. The decision threshold is fixed at 0.5. Tuning it on validation data is a possible later improvement.

**Promoting a new baseline is manual.** `gate/promote.py` runs as a dry run by default and only writes `baseline.json` with `--apply`. A candidate is eligible only if it passes the gate and also has at least 3 fewer missed defects than the baseline, so an improvement has to be bigger than the noise margin. Review the diff and commit it on its own. Never run it from CI.

**Tests.** [`tests/test_gate_checks.py`](tests/test_gate_checks.py) covers the check logic: a candidate exactly at the limit passes, one over it fails, and each limit is attributed to the right rule.

---

## Stage 5: CI/CD with GitHub Actions

Workflow: [`.github/workflows/ci.yml`](.github/workflows/ci.yml). It runs on pushes to `main` and on pull requests against it. A newer push to the same pull request cancels the run in progress.

Two jobs, in order:

1. **lint-and-test** runs `ruff check .` and `pytest` on Python 3.12. The second job only starts if this one passes.
2. **pipeline** runs the whole thing on a GitHub-hosted runner:
   - Installs the CPU build of PyTorch, then the pinned requirements.
   - Points DVC at the DagsHub remote using the `DAGSHUB_TOKEN` repository secret, and caches `.dvc/cache` keyed on the `.dvc` files so unchanged data isn't downloaded again.
   - Pulls the test set and the three batches, and rebuilds `data/processed/train` from the batches.
   - Trains on `--version v3`, evaluates that run on the test set, and runs the gate against `gate/baseline.json`.
   - Writes the gate report into the job summary, even when an earlier step failed.

If the gate fails or errors, the build fails. That's the point of the whole exercise: a model that regresses past the baseline can't get through a pull request unnoticed.

The training step has a 45-minute timeout, the evaluation step 15 minutes and the gate 5.

DagsHub was chosen as the remote because it's free and needed the least setup. The local remote is still the default in `.dvc/config`, which is why CI passes `-r dagshub` explicitly.

---

## Repo layout

```
.github/workflows/
  ci.yml            # lint, tests, then train -> evaluate -> gate
data/
  raw/              # original Kaggle download (local only, not versioned)
  processed/        # leakage-free, DVC-tracked: test, batch_1/2/3 (and a local train/)
  val_manifest.txt  # fixed validation split, tracked in Git
gate/
  baseline.json     # current baseline and gate policy
  common.py         # shared checks and loading logic
  gate.py           # the gate itself
  promote.py        # manual baseline promotion (dry run by default)
notebooks/
  01_eda.ipynb
  02_dataset_sanity_check.ipynb
  03_model_sanity_check.ipynb
scripts/
  find_leakage.py   # content-hash train/test and within-split duplicate detection
  clean_splits.py   # builds data/processed/ with leaked test images removed
  make_batches.py   # splits train into 3 stratified batches for DVC
  make_val_split.py # creates data/val_manifest.txt
src/
  config.py         # TrainConfig and the v1/v2/v3 data versions
  dataset.py        # CastingDataset and manifest reading
  model.py          # ResNet18 with a frozen backbone
  train.py          # training loop and MLflow logging
  evaluate.py       # one-shot test-set evaluation
tests/
  test_gate_checks.py
reports/
  figures/          # EDA plots (class balance, pixel intensity, sample grid)
  leaked_files.txt  # audit trail of removed leaked images
models/             # empty; trained models are stored as MLflow artifacts
requirements.txt, requirements-dev.txt, pytest.ini, ruff.toml
```

---

Stages 6 and 7 (serving and drift monitoring) get their own sections here once they're done.

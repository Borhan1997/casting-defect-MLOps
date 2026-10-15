# casting-defect-MLOps

Most of the work in a production ML system isn't the model. This repo builds the machinery around one: versioned data, reproducible training, an evaluation gate that can block a bad model, CI that runs all of it on every change, a containerized API, and drift monitoring on the traffic it serves.

The task itself is simple: spot defective metal castings in photos. The pipeline around it is the interesting part.

The project was built in seven stages, and each one is written up here. All seven are done.

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
| Serving | FastAPI |
| Containerization | Docker, GHCR |
| Monitoring | Prometheus, Grafana, plus a PSI drift script |

## Stage plan

| Stage | Scope | Status |
|---|---|---|
| 1 | Repo scaffolding, EDA, leakage check | Done |
| 2 | Data versioning with DVC, simulated time-batched data arrival | Done |
| 3 | PyTorch training pipeline (ResNet18) and MLflow tracking | Done |
| 4 | Evaluation gate against a stored baseline | Done |
| 5 | GitHub Actions CI/CD | Done |
| 6 | FastAPI serving, Docker image, GHCR | Done |
| 7 | Prediction logging, Prometheus and Grafana, drift detection | Done |

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

To run the API and the monitoring stack, skip to the end of Stage 6 and Stage 7.

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


---

## Stage 6: Serving

Code: [`src/api.py`](src/api.py), [`src/inference.py`](src/inference.py), [`Dockerfile`](Dockerfile)

**The API.** A small FastAPI app. The model loads once at startup, not per request.
- `GET /health` returns `{"status": "ok"}`.
- `POST /predict` takes an image as a multipart file upload named `file` and returns `{"label": "def_front", "probability": 0.97}`. The probability is the model's P(defect), and the label comes from a fixed 0.5 threshold, the same one the gate uses.
- An empty upload gets a 400 (`Empty file.`), and so does a file that isn't a readable image.

```bash
curl -F "file=@tests/fixtures/def_front.jpg" http://localhost:8000/predict
```

**Inference code.** `src/inference.py` has `load_model()` (cached) and `predict(image_bytes)`, and imports nothing from the training code. That keeps the serving image small. It applies the same torchvision ResNet18 preprocessing as training.

**The model file.** `models/baseline.pt` is a plain `state_dict`, tracked by DVC (`models/baseline.pt.dvc`) and stored on DagsHub like the data. `scripts/export_model.py` is the script that exports it. Pull it with:

```bash
dvc pull -r dagshub models/baseline.pt.dvc
```

**The image.** `python:3.12-slim`, CPU-only PyTorch with pinned versions (`requirements-serving.txt`), and a non-root user (uid 1000). Only the serving files and the model are copied in. Dependencies are installed before the code, so a code change doesn't reinstall PyTorch.

```bash
docker build -t casting-defect-api .
docker run --rm -p 8000:8000 casting-defect-api
```

**CI.** A third job, **serve**, runs after the pipeline job passes. It pulls the model, builds the image, starts it and smoke-tests it: the health check, a prediction, and (since Stage 7) the `/metrics` endpoint. On pushes to `main` it then pushes the image to GHCR as `ghcr.io/borhan1997/casting-defect-mlops`, tagged `latest` and `sha-<commit>`. Pull requests build and test the image but don't publish it. I checked that the published image gives the same answers as the local one.

**Tests.** [`tests/test_api.py`](tests/test_api.py) uses FastAPI's `TestClient` with the two fixture images in `tests/fixtures/`. The module is skipped when `models/baseline.pt` isn't there, so `pytest` still works on a clone that hasn't pulled the model.

---

## Stage 7: Prediction logging and drift monitoring

Training data is a snapshot. A deployed model sees whatever the camera sends next, and nobody labels it, so accuracy can't be tracked. What can be tracked is whether the inputs and the model's outputs still look like they did on the data the model was checked against.

Dashboard (the demo run described below):

![Grafana dashboard](docs/images/dashboard.png)

### What gets recorded

**Prediction log.** Off by default. If `PREDICTION_LOG_PATH` is set, every served prediction appends one line of JSON to that file: timestamp, request ID, label, probability, image width and height, and three image statistics. No images are stored. Rejected requests (empty or unreadable files) aren't logged. The logger never raises, so a full disk can't turn into a failed prediction. Code: [`src/monitoring.py`](src/monitoring.py).

**Image statistics.** Computed on the original image, before any resizing:
- brightness: mean of the grayscale image
- contrast: standard deviation of the grayscale image
- sharpness: variance of the Laplacian (low means blurry)

**Prometheus metrics.** `GET /metrics` exposes a prediction counter by label, a latency histogram, a histogram of the defect probability, and histograms of the three image statistics. Code: [`src/metrics.py`](src/metrics.py).

### The reference profile

Drift only means something against a reference. [`src/build_reference.py`](src/build_reference.py) writes `monitoring/reference_profile.json`:
- **Input statistics** come from the 5638 training images the baseline model was trained on (train minus the 995 validation images). The test set is left alone.
- **Prediction statistics** come from running the model on the 995 validation images: defect rate 0.565, mean probability 0.568, and a 10-bin histogram of the probability. About 83% of the probability mass sits in the two outer bins, so the model is usually confident.

The file is committed to Git and copied into the Docker image. The histogram buckets of the image statistics use the deciles of the reference, so the dashboard and the drift script look at the same bins. To rebuild it (only needed if the baseline model changes):

```bash
python -m src.build_reference --images-dir data/processed/train \
  --val-manifest data/val_manifest.txt --predictions
```

### Running the stack

```bash
mkdir -p logs        # create it yourself, see below
docker compose up -d
```

| Service | URL | Notes |
|---|---|---|
| API | http://localhost:8000 | writes `logs/predictions.jsonl` |
| Prometheus | http://localhost:9090 | scrapes `/metrics` every 5 seconds |
| Grafana | http://localhost:3000 | login `admin` / `admin` |

Create `logs/` before the first `up`. If Docker creates it, it belongs to root and the container user (uid 1000) can't write to it.

Grafana comes up already configured: the Prometheus datasource and the dashboard are provisioned from files under `monitoring/grafana/provisioning/`, so there's nothing to click through. The dashboard (uid `casting-defect`) has three rows:
- **Prediction drift:** defect rate, mean probability, and the share of uncertain predictions (probability between 0.1 and 0.9).
- **Input drift:** mean brightness, contrast and sharpness.
- **Service health:** p95 latency.

Each drift panel draws a dashed line at the reference value, so you can see the gap without reading numbers. Those lines are fixed numbers written into the dashboard JSON when it was generated. If you rebuild the reference profile, they need regenerating too.

### The drift script

The dashboard is for looking. [`src/drift.py`](src/drift.py) is for deciding. It reads the last N records of the log and scores each signal against the reference with the population stability index (PSI):

- Brightness, contrast and sharpness are binned into the reference deciles. Values outside the reference range are clipped into the outer bins.
- The probability is binned into three groups: below 0.1, 0.1 to 0.9, and 0.9 or above.
- Empty bins are floored at 1e-4 so the log is defined.
- Under 0.1 is stable, 0.1 to 0.25 is moderate, 0.25 and up is major. These are the usual rules of thumb for PSI, not something tuned on this data.

```bash
python -m src.drift                     # last 200 records
python -m src.drift --window 300 --json-out drift.json
```

It prints a table and exits with `0` (nothing major), `1` (at least one major signal) or `2` (not enough data, or the log can't be read). The minimum is 200 records. PSI on a small window is noisy: in a simulation on clean traffic, a 100-record window read "moderate" on at least one signal about three times in four, and 200 records cut that to roughly one in thirteen with "major" practically never. So the script refuses to score fewer.

### Demo: making the drift show up

[`src/simulate_drift.py`](src/simulate_drift.py) replays images against the running API, 300 requests at 3 per second per phase, and each phase was scored with `python -m src.drift --window 300` on a fresh log:

- **clean:** validation images as they are. The same distribution as the reference.
- **natural:** the test images as they are.
- **dark:** the same validation images with brightness halved.
- **blur:** the same validation images with a Gaussian blur of radius 2.

```bash
python -m src.simulate_drift                    # all four phases, in order
python -m src.simulate_drift --phases dark      # or just one
```

| Phase | Defect rate | Brightness PSI | Contrast PSI | Sharpness PSI | Probability PSI | Overall |
|---|---|---|---|---|---|---|
| clean | 0.580 | 0.019 | 0.019 | 0.036 | 0.004 | stable |
| natural | 0.713 | 0.178 | 0.126 | 0.334 | 0.081 | major |
| dark | 0.830 | 8.283 | 8.283 | 7.911 | 1.579 | major |
| blur | 0.443 | 0.021 | 1.369 | 8.283 | 0.464 | major |

(The reference defect rate is 0.565. 8.283 is the highest PSI the script can return with these settings: every record falls outside the reference range.)

What these runs show:
- **Clean traffic stays quiet,** which is the baseline that makes the other rows mean something.
- **The two faults push the defect rate in opposite directions.** Dark images raised it from 0.58 to 0.83, and blurred ones lowered it to 0.443. There are no labels in this setup, so this is inference: the dark images are the same ones the clean phase scored, so the extra defects are probably false alarms, and the blurred images probably hide real defects. I didn't measure that.
- **Contrast flagged the blur** (PSI 1.37) even though brightness didn't move, because the reference contrast spread is narrow (a standard deviation of about 3.5 across images).
- **The natural phase is flagged as major, yet the model is fine on that data.** The test set is sharper than the training set (mean sharpness 236 against 183), and the baseline still gets 98.3% accuracy on it. Drift means the inputs changed. It doesn't mean the model got worse, and only labels can tell you that.

### Limitations

- **The defect rate is reported but not scored.** On the natural phase it moved by 15 points, but the 3-bin probability PSI stayed at 0.081 ("stable"), so a shift in how many defects come in can pass unnoticed. The script prints the gap next to the table; it doesn't count toward the overall verdict.
- **The drift script is run by hand.** Nothing schedules it and nothing alerts. Prometheus and Grafana only display.
- **It's all local.** The compose stack runs on one machine. The CI job builds and publishes the image, but nothing deploys it.
- **Prometheus and Grafana keep no data between runs.** There are no volumes for them, so `docker compose down` clears the history. The prediction log survives because `logs/` is a bind mount.
- **PSI is only as good as its window.** Under 200 records the result is mostly noise (see above).
- **No labels, no accuracy.** Everything in this stage is a proxy for the model going wrong.

### Tests

New test files cover the logger ([`test_monitoring.py`](tests/test_monitoring.py), [`test_api_logging.py`](tests/test_api_logging.py)), the reference builder, the metrics (including `/metrics` through the API), the drift scoring and the simulator. The full suite is 104 tests.

---

## Repo layout

```
.github/workflows/
  ci.yml                    # lint and tests, train -> evaluate -> gate, then build/test/push the image
data/
  raw/                      # original Kaggle download (local only, not versioned)
  processed/                # leakage-free, DVC-tracked: test, batch_1/2/3 (and a local train/)
  val_manifest.txt          # fixed validation split, tracked in Git
docs/
  images/                   # dashboard screenshot used above
gate/
  baseline.json             # current baseline and gate policy
  common.py                 # shared checks and loading logic
  gate.py                   # the gate itself
  promote.py                # manual baseline promotion (dry run by default)
logs/                       # prediction log from docker compose (not in Git)
models/
  baseline.pt               # serving model, a state_dict tracked by DVC (baseline.pt.dvc)
monitoring/
  prometheus.yml            # scrape config
  reference_profile.json    # reference statistics, tracked in Git, copied into the image
  grafana/provisioning/     # datasource and dashboard, loaded when Grafana starts
notebooks/
  01_eda.ipynb
  02_dataset_sanity_check.ipynb
  03_model_sanity_check.ipynb
scripts/
  find_leakage.py           # content-hash train/test and within-split duplicate detection
  clean_splits.py           # builds data/processed/ with leaked test images removed
  make_batches.py           # splits train into 3 stratified batches for DVC
  make_val_split.py         # creates data/val_manifest.txt
  export_model.py           # exports the trained model to models/baseline.pt
src/
  config.py                 # TrainConfig and the v1/v2/v3 data versions
  dataset.py                # CastingDataset and manifest reading
  model.py                  # ResNet18 with a frozen backbone
  train.py                  # training loop and MLflow logging
  evaluate.py               # one-shot test-set evaluation
  api.py                    # FastAPI app (/health, /predict, /metrics)
  inference.py              # model loading and prediction, no training imports
  monitoring.py             # prediction logging and image statistics
  metrics.py                # Prometheus metrics
  build_reference.py        # writes monitoring/reference_profile.json
  drift.py                  # PSI drift report over the prediction log
  simulate_drift.py         # replays clean, natural, dark and blurred traffic
tests/
  fixtures/                 # one defective and one OK image
  test_gate_checks.py
  test_api.py
  test_api_logging.py
  test_api_metrics.py
  test_monitoring.py
  test_metrics.py
  test_build_reference.py
  test_drift.py
  test_simulate_drift.py
reports/
  figures/                  # EDA plots (class balance, pixel intensity, sample grid)
  leaked_files.txt          # audit trail of removed leaked images
Dockerfile, .dockerignore, docker-compose.yml
requirements.txt, requirements-dev.txt, requirements-serving.txt, pytest.ini, ruff.toml
```
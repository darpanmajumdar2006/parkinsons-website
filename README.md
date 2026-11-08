# Parkinson's Speech-Feature ML Demo

A web app and API for experimenting with a binary classifier trained on
precomputed Parkinson's speech-feature data. It accepts numeric feature values;
it does not record, upload, or analyze audio.

**Demo:** [parkinsons-mlops.onrender.com](https://parkinsons-mlops.onrender.com)

> **Research and educational use only.** This project is not a medical device
> and must not be used to diagnose, treat, or make health decisions.

## What it does

- Accepts a CSV containing named speech-feature columns or manual numeric input.
- Lets you select a row from a CSV and displays the predicted class and model
  probabilities.
- Shows the Random Forest's held-out accuracy, precision, recall, F1 score,
  ROC-AUC, and confusion matrix on the homepage.
- Trains a scikit-learn pipeline using a stratified holdout split, then saves a
  model for serving.
- Serves a static website and JSON API from FastAPI.

The model expects all **753 numeric features**. The feature names are available
from `GET /schema`. The web app guides CSV and manual input; raw audio is not
supported.

## Run locally

Use Python 3.12 or newer. Run these commands from the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). To run the tests, open
another terminal in the project directory, activate the environment, and run:

```powershell
python -m pytest -q tests
```

Training reads `data/pd_dataset.csv`, drops rows with missing values, excludes
identifier columns, and uses the `class` column as the target. It evaluates a
stratified 20% holdout split, then fits the serving model on all clean rows.
The model and holdout metrics are written to:

- `models/parkinsons_model.joblib`
- `models/metrics.json`

The dataset has a grouped header row, which the loader accounts for. The
classifier is a class-balanced random forest with 200 trees.

## Website and API

| Route | Purpose |
| --- | --- |
| `/` | Prediction website |
| `/privacy` | Privacy information |
| `/health` | Service health check |
| `/schema` | Required model feature names |
| `/evaluation` | Saved held-out metrics and confusion matrix |
| `/predict` | JSON prediction endpoint |
| `/explain` | SHAP feature contributions for one prediction |
| `/robots.txt` | Crawler rules |
| `/sitemap.xml` | Public website URLs |

Interactive API documentation and the OpenAPI schema are intentionally
disabled: `/docs`, `/redoc`, and `/openapi.json` return 404. This only hides
documentation; it does **not** disable or protect the API. The service is public
and has no authentication, so anyone with its URL can submit predictions.

`GET /evaluation` returns the metrics, confusion matrix, class labels, and test
split fraction calculated during training. These metrics use a stratified
20% holdout split and are distinct from the probabilities returned for an
individual prediction.

`POST /predict` accepts a JSON object with every feature returned by `/schema`,
mapped to finite numbers. Missing, unexpected, or non-finite values are
rejected. For example, the request has this shape (the abbreviated feature
object is not sufficient for a real request):

```json
{
  "features": {
    "gender": 1,
    "PPE": 0.75
  }
}
```

The response contains a `prediction` (`0` or `1`) and a `probabilities` object
keyed by class. These probabilities are model outputs, not medical confidence
or clinical evidence.

After a prediction, the website requests a local SHAP explanation from
`POST /explain` using the same feature values. The response includes the
predicted class probability, its baseline, and the eight largest feature
contributions. SHAP values are expressed in probability points; the full set of
feature contributions sums with the baseline to the predicted class
probability. They explain this model's output only; they do not identify
medical causes or provide a diagnosis.

The explanation is calculated on the server with SHAP's `TreeExplainer` for the
Random Forest. It is a local, per-input explanation; it does not change the
model's prediction. If explanation generation is unavailable, the prediction
still appears and the explanation panel reports that it could not be loaded.

## Deploy to Render

The `render.yaml` Blueprint defines a free Python web service. It installs
dependencies, trains the model during the build, and starts the API with
Uvicorn. To deploy:

1. Push this project to a GitHub repository.
2. In Render, choose **New → Blueprint** and connect that repository.
3. Select this project's `render.yaml` and deploy the service.
4. Open the service URL and check `/health`. The default URL configured in this
   project is [https://parkinsons-mlops.onrender.com](https://parkinsons-mlops.onrender.com).

Free Render services may spin down while idle, so the first request after a
period of inactivity may take longer.

### Deployment settings

- `SITE_URL` is the canonical public HTTP(S) origin used for metadata, robots,
  and the sitemap. Update it if the deployed URL or custom domain differs.
- `PLAUSIBLE_DOMAIN` controls the optional Plausible Analytics script. The
  Blueprint sets it to the default service domain; register that domain with
  Plausible to receive reports. Leave it unset to disable the script.
- `MODEL_PATH` optionally overrides the model artifact path. By default, the
  app loads `models/parkinsons_model.joblib`.
- Render provides `PORT` to Uvicorn.

## Run with Docker

The Docker image installs dependencies and trains the model from the bundled
dataset during its build:

```powershell
docker build -t parkinsons-mlops .
docker run --rm -p 8000:8000 parkinsons-mlops
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The container uses port
`8000` by default; `PORT` and `MODEL_PATH` can be overridden when running it.

## Project structure

```text
parkinsons-mlops/
├── api/
│   ├── main.py
│   └── static/          # Website, privacy page, 404 page, and image assets
├── data/
│   └── pd_dataset.csv
├── models/              # Generated model and training metrics
├── src/
│   ├── preprocessing.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   ├── test_train.py
│   └── test_explainability.py
├── Dockerfile
├── render.yaml
└── requirements.txt
```

# Parkinson's Speech-Feature MLOps Service

Train and serve a binary classifier using the precomputed speech features in
`data/pd_dataset.csv`. This project accepts feature values; it does not process
raw audio recordings.
It is a demonstration classifier, not a medical diagnostic tool.

**Live demo:** [parkinsons-mlops.onrender.com](https://parkinsons-mlops.onrender.com)

`data analysis.py` is included unchanged as the original exploratory script;
the copy in the neighboring `parkinson aiml` folder is also left untouched.
The deployment uses the separate training and serving code in `src/` and `api/`.

## Project layout

```text
parkinsons-mlops/
├── data/
├── models/
├── src/
│   ├── preprocessing.py
│   └── train.py
├── api/
│   ├── main.py
│   └── static/
│       ├── index.html
│       ├── privacy.html
│       ├── 404.html
│       └── site assets
├── tests/
│   └── test_api.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Run locally

Use Python 3.12 or newer. From this directory:

```bash
python -m venv .venv
```

Activate the virtual environment, then install dependencies and train:

```bash
python -m pip install -r requirements.txt
python -m src.train
pytest -q
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Training writes the deployable model to `models/parkinsons_model.joblib` and
holdout metrics to `models/metrics.json`. The model is evaluated on a
stratified 20% holdout split, then refit on all clean rows for serving.
Identifiers are excluded from model features; the `gender` feature is retained.

## API

- `GET /` — interactive feature-analysis frontend.
- `GET /privacy` — privacy policy for prediction input and optional analytics.
- `GET /robots.txt` and `GET /sitemap.xml` — crawler guidance and public page URLs.
- `GET /health` — liveness check.
- `GET /schema` — lists all required feature names.
- `POST /predict` — returns the predicted class and per-class probabilities.

Interactive API documentation and the OpenAPI schema are intentionally disabled
on the public service. `/docs`, `/redoc`, and `/openapi.json` return 404. The
prediction and schema endpoints remain available to the website and API clients.

Open the frontend at `/`. Upload a feature CSV with named columns and one or
more rows, choose a row, and run a prediction; identifier and target columns
are ignored. The alternate manual-entry mode exposes searchable, grouped
fields, but still requires all 753 numeric model features. Raw audio is not
accepted or analyzed.

The prediction body must contain every feature returned by `/schema`, each
mapped to a finite numeric value. Missing or unknown feature names are rejected.
The abbreviated request below illustrates the JSON shape; a real request must
include every schema feature:

```json
{
  "features": {
    "gender": 1,
    "PPE": 0.75
  }
}
```

The example is abbreviated: include all schema features in a real prediction.
The response contains a `prediction` (`0` or `1`) and a `probabilities` map.

## Build and run with Docker

Docker builds install the dependencies and train the model from the bundled
CSV, so a model artifact does not need to be committed:

```bash
docker build -t parkinsons-mlops .
docker run --rm -p 8000:8000 parkinsons-mlops
```

The service listens on port `8000`. Configure `PORT` to use a different
container port. The model path can be overridden with `MODEL_PATH` when
running the API outside the provided Docker image.

## Deploy publicly on Render

The `render.yaml` Blueprint deploys this API as a Python web service and trains
the model during the build. It uses Render's Free plan and does not require
Docker. Free services can spin down when idle and may take about a minute to
wake on the next request.

1. Put this project in its own GitHub repository. Do not push it to the
   unrelated repository in the parent workspace.
2. In Render, choose **New → Blueprint** and connect that repository.
3. Select `render.yaml` when prompted, review the service settings, and deploy.
4. Open the Render service URL and use `/health` to check that the service is
   responding. The default service URL is
   `https://parkinsons-mlops.onrender.com/`; interactive API documentation is
   disabled on the public service.

The API is public and does not include authentication. Anyone with its URL can
submit predictions. Do not use it as a medical diagnostic tool.

## Website metadata and analytics

The site includes page-specific titles and descriptions, Open Graph/Twitter share
metadata, a branded share image and favicon assets, a custom 404 page, and
accessible loading feedback for schema loading and predictions. The homepage CTA
links directly to the analysis workspace. Decorative in-page illustrations are
hidden from assistive technology; the social share image has descriptive
alternative metadata.

The Render Blueprint sets `SITE_URL` to the service's public URL and enables
Plausible Analytics for that domain. Register the same domain in your Plausible
account to view aggregate visit reports. For a custom domain or another
deployment, update `SITE_URL` to the canonical HTTPS origin and set
`PLAUSIBLE_DOMAIN` to the registered Plausible site domain. Leave
`PLAUSIBLE_DOMAIN` unset to disable the analytics script. The privacy policy is
available at `/privacy`.

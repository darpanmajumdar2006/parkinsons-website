from functools import lru_cache
from html import escape
import math
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator


STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "parkinsons_model.joblib"
)

app = FastAPI(
    title="Parkinson's Speech Feature Classifier",
    description="Predicts a class from precomputed speech-feature values.",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(min_length=1)

    @field_validator("features")
    @classmethod
    def values_must_be_finite(cls, features: dict[str, float]) -> dict[str, float]:
        if not all(math.isfinite(value) for value in features.values()):
            raise ValueError("Feature values must be finite numbers.")
        return features


class PredictionResponse(BaseModel):
    prediction: int
    probabilities: dict[str, float]


class EvaluationResponse(BaseModel):
    metrics: dict[str, float]
    confusion_matrix: list[list[int]]
    class_labels: list[int]
    test_size: float


class FeatureContribution(BaseModel):
    feature_name: str
    feature_value: float
    shap_value: float


class ExplanationResponse(BaseModel):
    prediction: int
    class_probability: float
    base_value: float
    contributions: list[FeatureContribution]


@lru_cache(maxsize=1)
def load_model_bundle() -> dict[str, Any]:
    model_path = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model artifact not found at {model_path}. "
                "Train the model before starting the API."
            ),
        )
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_shap_explainer() -> Any:
    bundle = load_model_bundle()
    model = bundle["model"]
    return shap.TreeExplainer(model.named_steps["model"])


def explain_prediction(
    bundle: dict[str, Any],
    input_frame: pd.DataFrame,
    predicted_class: int,
    class_probability: float,
) -> ExplanationResponse:
    model = bundle["model"]
    class_labels = [int(label) for label in model.classes_]
    try:
        class_index = class_labels.index(predicted_class)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail="The model returned an unknown class for its explanation.",
        ) from exc

    scaled_features = model.named_steps["scaler"].transform(input_frame)
    explainer = load_shap_explainer()
    raw_values = explainer.shap_values(scaled_features)
    if isinstance(raw_values, list):
        class_values = np.asarray(raw_values[class_index])[0]
    else:
        values = np.asarray(raw_values)
        if values.ndim == 3 and values.shape[0] == 1:
            class_values = values[0, :, class_index]
        else:
            raise HTTPException(
                status_code=503,
                detail="The SHAP explainer returned an unsupported output shape.",
            )

    expected_values = np.asarray(explainer.expected_value).reshape(-1)
    if class_index >= len(expected_values):
        raise HTTPException(
            status_code=503,
            detail="The SHAP explainer did not return a baseline for this class.",
        )
    base_value = float(expected_values[class_index])
    if not math.isclose(
        base_value + float(np.sum(class_values)),
        class_probability,
        rel_tol=1e-4,
        abs_tol=1e-4,
    ):
        raise HTTPException(
            status_code=503,
            detail="SHAP contributions did not match the model output.",
        )

    feature_values = input_frame.iloc[0].to_numpy(dtype=float)
    feature_names = bundle["feature_names"]
    top_indexes = np.argsort(np.abs(class_values))[-8:][::-1]
    contributions = [
        FeatureContribution(
            feature_name=feature_names[index],
            feature_value=float(feature_values[index]),
            shap_value=float(class_values[index]),
        )
        for index in top_indexes
    ]
    return ExplanationResponse(
        prediction=predicted_class,
        class_probability=class_probability,
        base_value=base_value,
        contributions=contributions,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def public_site_url() -> str:
    site_url = os.environ.get(
        "SITE_URL", "https://parkinsons-mlops.onrender.com"
    ).rstrip("/")
    parsed_url = urlsplit(site_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise RuntimeError("SITE_URL must be an absolute HTTP or HTTPS URL.")
    return site_url


def render_page(filename: str) -> HTMLResponse:
    site_url = public_site_url()
    plausible_domain = os.environ.get("PLAUSIBLE_DOMAIN", "").strip()
    analytics_script = ""
    if plausible_domain:
        analytics_script = (
            '<script defer data-domain="'
            + escape(plausible_domain, quote=True)
            + '" src="https://plausible.io/js/script.js"></script>'
        )

    page = (STATIC_DIR / filename).read_text(encoding="utf-8")
    page = page.replace("__SITE_URL__", escape(site_url, quote=True))
    page = page.replace("<!-- PLAUSIBLE_ANALYTICS -->", analytics_script)
    return HTMLResponse(page)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage() -> HTMLResponse:
    return render_page("index.html")


@app.get("/privacy", response_class=HTMLResponse, include_in_schema=False)
def privacy_policy() -> HTMLResponse:
    return render_page("privacy.html")


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
def robots() -> PlainTextResponse:
    site_url = public_site_url()
    return PlainTextResponse(
        f"User-agent: *\nAllow: /\nSitemap: {site_url}/sitemap.xml\n"
    )


@app.get("/sitemap.xml", include_in_schema=False)
def sitemap() -> Response:
    site_url = public_site_url()
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"<url><loc>{escape(site_url)}/</loc></url>"
        f"<url><loc>{escape(site_url)}/privacy</loc></url>"
        "</urlset>"
    )
    return Response(content=body, media_type="application/xml")


@app.get("/schema")
def feature_schema() -> dict[str, list[str]]:
    bundle = load_model_bundle()
    return {"features": bundle["feature_names"]}


@app.get("/evaluation", response_model=EvaluationResponse)
def model_evaluation() -> EvaluationResponse:
    bundle = load_model_bundle()
    evaluation = bundle.get("evaluation")
    if not isinstance(evaluation, dict):
        raise HTTPException(
            status_code=503,
            detail="Model evaluation data is unavailable. Retrain the model.",
        )
    return EvaluationResponse.model_validate(evaluation)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    bundle = load_model_bundle()
    feature_names = bundle["feature_names"]
    supplied_names = set(request.features)
    expected_names = set(feature_names)
    missing = sorted(expected_names - supplied_names)
    extra = sorted(supplied_names - expected_names)
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    input_frame = pd.DataFrame(
        [[request.features[name] for name in feature_names]],
        columns=feature_names,
    )
    model = bundle["model"]
    prediction = int(model.predict(input_frame)[0])
    probabilities = model.predict_proba(input_frame)[0]
    class_probabilities = {
        str(int(label)): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    return PredictionResponse(
        prediction=prediction,
        probabilities=class_probabilities,
    )


@app.post("/explain", response_model=ExplanationResponse)
def explain(request: PredictionRequest) -> ExplanationResponse:
    bundle = load_model_bundle()
    feature_names = bundle["feature_names"]
    supplied_names = set(request.features)
    expected_names = set(feature_names)
    missing = sorted(expected_names - supplied_names)
    extra = sorted(supplied_names - expected_names)
    if missing or extra:
        raise HTTPException(
            status_code=422,
            detail={"missing_features": missing, "unexpected_features": extra},
        )

    input_frame = pd.DataFrame(
        [[request.features[name] for name in feature_names]],
        columns=feature_names,
    )
    model = bundle["model"]
    probabilities = model.predict_proba(input_frame)[0]
    prediction = int(model.predict(input_frame)[0])
    class_index = [int(label) for label in model.classes_].index(prediction)
    return explain_prediction(
        bundle,
        input_frame,
        prediction,
        float(probabilities[class_index]),
    )


app.mount(
    "/",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="frontend",
)

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "parkinsons_model.joblib"
)

app = FastAPI(
    title="Parkinson's Speech Feature Classifier",
    description="Predicts a class from precomputed speech-feature values.",
    version="1.0.0",
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/schema")
def feature_schema() -> dict[str, list[str]]:
    bundle = load_model_bundle()
    return {"features": bundle["feature_names"]}


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

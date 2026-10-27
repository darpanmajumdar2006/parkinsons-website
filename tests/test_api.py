import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier

from api import main


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0, 1]))
    bundle = {"feature_names": ["feature_a", "feature_b"], "model": model}
    monkeypatch.setattr(main, "load_model_bundle", lambda: bundle)
    return TestClient(main.app)


def test_health(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_schema_returns_required_feature_names(client: TestClient) -> None:
    response = client.get("/schema")

    assert response.status_code == 200
    assert response.json() == {"features": ["feature_a", "feature_b"]}


def test_predict_returns_class_and_probabilities(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"features": {"feature_a": 0.25, "feature_b": 0.75}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "probabilities": {"0": 0.0, "1": 1.0},
    }


def test_predict_rejects_missing_or_unexpected_features(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"features": {"feature_a": 0.25, "unexpected": 0.75}},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == {
        "missing_features": ["feature_b"],
        "unexpected_features": ["unexpected"],
    }


def test_prediction_request_rejects_non_finite_feature_values() -> None:
    with pytest.raises(ValidationError):
        main.PredictionRequest(
            features={"feature_a": float("nan"), "feature_b": 0.75}
        )

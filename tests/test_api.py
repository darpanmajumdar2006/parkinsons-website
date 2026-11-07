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


def test_frontend_is_served_at_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "<title>Parkinson's ML Studio | VoiceSignal</title>" in response.text
    assert 'name="description"' in response.text
    assert 'property="og:image"' in response.text
    assert 'property="og:image:alt"' in response.text
    assert 'href="#workspace"' in response.text
    assert "/styles.css" in response.text
    assert "/docs" not in response.text
    assert "Developer API" not in response.text


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_documentation_is_not_public(client: TestClient, path: str) -> None:
    response = client.get(path)

    assert response.status_code == 404


def test_privacy_page_and_custom_404_are_served(client: TestClient) -> None:
    privacy_response = client.get("/privacy")
    not_found_response = client.get("/a-page-that-does-not-exist")

    assert privacy_response.status_code == 200
    assert "<title>Privacy Policy | VoiceSignal</title>" in privacy_response.text
    assert 'name="description" content="Learn what information' in privacy_response.text
    assert "Plausible Analytics" in privacy_response.text
    assert not_found_response.status_code == 404
    assert "<title>Page not found | VoiceSignal</title>" in not_found_response.text
    assert 'name="description" content="The page you requested' in not_found_response.text


def test_search_metadata_uses_configured_public_url_and_analytics(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SITE_URL", "https://demo.example")
    monkeypatch.setenv("PLAUSIBLE_DOMAIN", "demo.example")

    response = client.get("/")

    assert response.status_code == 200
    assert '<link rel="canonical" href="https://demo.example/">' in response.text
    assert 'property="og:image" content="https://demo.example/og-image.png"' in response.text
    assert 'data-domain="demo.example"' in response.text
    assert 'src="https://plausible.io/js/script.js"' in response.text


def test_robots_and_sitemap_use_configured_public_url(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SITE_URL", "https://demo.example")

    robots_response = client.get("/robots.txt")
    sitemap_response = client.get("/sitemap.xml")

    assert robots_response.status_code == 200
    assert "Sitemap: https://demo.example/sitemap.xml" in robots_response.text
    assert sitemap_response.status_code == 200
    assert sitemap_response.headers["content-type"].startswith("application/xml")
    assert "<loc>https://demo.example/</loc>" in sitemap_response.text
    assert "<loc>https://demo.example/privacy</loc>" in sitemap_response.text


def test_frontend_assets_are_served(client: TestClient) -> None:
    css_response = client.get("/styles.css")
    js_response = client.get("/app.js")

    assert css_response.status_code == 200
    assert "--green-deep" in css_response.text
    assert js_response.status_code == 200
    assert 'fetch("/predict"' in js_response.text
    for asset in (
        "/favicon.svg",
        "/favicon-16x16.png",
        "/favicon-32x32.png",
        "/apple-touch-icon.png",
        "/android-chrome-192x192.png",
        "/android-chrome-512x512.png",
        "/site.webmanifest",
    ):
        assert client.get(asset).status_code == 200
    og_image_response = client.get("/og-image.png")
    assert og_image_response.status_code == 200
    assert og_image_response.headers["content-type"] == "image/png"


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

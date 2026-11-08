import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from api import main


class FakeTreeExplainer:
    expected_value = np.array([0.3, 0.7])

    def shap_values(self, values: np.ndarray) -> np.ndarray:
        assert values.shape == (1, 2)
        return np.array([[[0.1, 0.1], [-0.1, 0.2]]])


def test_explanation_returns_top_features_for_predicted_class(
    monkeypatch,
) -> None:
    classifier = DummyClassifier(strategy="constant", constant=1)
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )
    model.fit(
        pd.DataFrame({"feature_a": [0.0, 1.0], "feature_b": [1.0, 3.0]}),
        np.array([0, 1]),
    )
    bundle = {
        "model": model,
        "feature_names": ["feature_a", "feature_b"],
    }
    frame = pd.DataFrame({"feature_a": [2.0], "feature_b": [3.0]})
    monkeypatch.setattr(main, "load_shap_explainer", FakeTreeExplainer)

    explanation = main.explain_prediction(
        bundle,
        frame,
        predicted_class=1,
        class_probability=1.0,
    )

    assert explanation.prediction == 1
    assert explanation.class_probability == 1.0
    assert explanation.base_value == 0.7
    assert [
        (item.feature_name, item.feature_value, item.shap_value)
        for item in explanation.contributions
    ] == [
        ("feature_b", 3.0, 0.2),
        ("feature_a", 2.0, 0.1),
    ]

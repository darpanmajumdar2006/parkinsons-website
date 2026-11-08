import json

import joblib
import numpy as np
import pandas as pd

from src import train as training


def test_training_persists_holdout_metrics_and_confusion_matrix(
    tmp_path, monkeypatch
) -> None:
    labels = np.repeat([0, 1], 20)
    dataframe = pd.DataFrame(
        {
            "feature_a": np.concatenate((np.zeros(20), np.ones(20))),
            "feature_b": np.arange(40, dtype=float),
            "class": labels,
        }
    )
    monkeypatch.setattr(
        training,
        "load_dataset",
        lambda dataset_path: dataframe,
    )
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"

    metrics = training.train(
        dataset_path=tmp_path / "unused.csv",
        model_path=model_path,
        metrics_path=metrics_path,
    )

    artifact = joblib.load(model_path)
    evaluation = artifact["evaluation"]
    persisted_evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert evaluation["metrics"] == metrics
    assert evaluation["class_labels"] == [0, 1]
    assert evaluation["test_size"] == 0.2
    assert len(evaluation["confusion_matrix"]) == 2
    assert all(len(row) == 2 for row in evaluation["confusion_matrix"])
    assert sum(map(sum, evaluation["confusion_matrix"])) == 8
    assert persisted_evaluation == evaluation

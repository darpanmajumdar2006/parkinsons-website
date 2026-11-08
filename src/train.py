import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import __version__ as sklearn_version

from src.preprocessing import load_dataset, prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "pd_dataset.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "parkinsons_model.joblib"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
TEST_SIZE = 0.2


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
) -> dict[str, float]:
    dataframe = load_dataset(dataset_path)
    features, target = prepare_features(dataframe)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=42,
        stratify=target,
    )

    evaluation_model = build_model()
    evaluation_model.fit(x_train, y_train)
    predictions = evaluation_model.predict(x_test)
    probabilities = evaluation_model.predict_proba(x_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }
    evaluation = {
        "metrics": metrics,
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
            labels=evaluation_model.classes_,
        ).tolist(),
        "class_labels": [int(label) for label in evaluation_model.classes_],
        "test_size": TEST_SIZE,
    }

    final_model = build_model()
    final_model.fit(features, target)
    artifact = {
        "artifact_version": 1,
        "model": final_model,
        "feature_names": features.columns.tolist(),
        "classes": [int(label) for label in final_model.classes_],
        "target_column": "class",
        "sklearn_version": sklearn_version,
        "metrics": metrics,
        "evaluation": evaluation,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)
    metrics_path.write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Parkinson's model.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-output", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    metrics = train(args.dataset, args.model_output, args.metrics_output)
    print("Holdout metrics:")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {args.model_output}")


if __name__ == "__main__":
    main()

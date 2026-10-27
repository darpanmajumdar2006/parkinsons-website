from pathlib import Path

import pandas as pd


TARGET_COLUMN = "class"
IDENTIFIER_COLUMNS = ("id", "subject_id", "subjectid", "subject")


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the speech-feature CSV, whose first row contains grouped headers."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    return pd.read_csv(dataset_path, header=1)


def prepare_features(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate the dataset and return numeric features and class labels."""
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(f"Dataset must contain a {TARGET_COLUMN!r} target column.")

    clean_dataframe = dataframe.dropna().copy()
    if clean_dataframe.empty:
        raise ValueError("Dataset has no rows after removing missing values.")

    target = clean_dataframe[TARGET_COLUMN]
    feature_columns = [
        column
        for column in clean_dataframe.columns
        if column != TARGET_COLUMN and column.lower() not in IDENTIFIER_COLUMNS
    ]
    features = clean_dataframe[feature_columns]

    non_numeric_columns = features.select_dtypes(exclude="number").columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            f"All model features must be numeric; invalid columns: "
            f"{non_numeric_columns}"
        )
    if target.nunique() != 2:
        raise ValueError("The target column must contain exactly two classes.")

    return features, target

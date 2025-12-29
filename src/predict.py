# predict.py

import pandas as pd
import joblib
from pathlib import Path


def predict_from_dataframe(df: pd.DataFrame, model_path: Path):
    """
    Predict churn for given dataframe
    df: DataFrame containing feature columns only (no target)
    """

    # Load trained pipeline
    pipeline = joblib.load(model_path)

    # Predictions
    predictions = pipeline.predict(df)

    # Probabilities (if supported)
    probabilities = None
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        probabilities = pipeline.predict_proba(df)[:, 1]

    return predictions, probabilities


if __name__ == "__main__":
    """
    This file is meant to be IMPORTED, not run directly.
    Example usage:
        from predict import predict_from_dataframe
    """
    pass


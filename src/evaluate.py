# evaluate.py

import pandas as pd
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)


def evaluate_model(data_path, model_path, target_col):
    # 1️⃣ Load data
    df = pd.read_csv(data_path)

    # Drop rows where target is missing (same rule as training)
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2️⃣ Load trained pipeline (preprocessing + model)
    pipeline = joblib.load(model_path)

    # 3️⃣ Predictions
    y_pred = pipeline.predict(X)

    # 4️⃣ Metrics
    print("\n📊 MODEL EVALUATION RESULTS")
    print("=" * 35)

    print("Accuracy:", accuracy_score(y, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))

    # 5️⃣ ROC-AUC (RandomForest supports predict_proba)
    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        y_prob = pipeline.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
        print("ROC-AUC:", auc)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw_data" / "customer_churn_dataset-training-master.csv"
MODEL_PATH = BASE_DIR / "models" /"churn_model_pipeline.joblib"

if __name__ == "__main__":
    evaluate_model(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        target_col="Churn"
    )

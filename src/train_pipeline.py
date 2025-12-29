# train.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path

from data_preprocessing import split_data, build_preprocessor


def train_model(data_path, target_col):
    #Load data
    df = pd.read_csv(data_path)

    #Train-Test Split
    X_train, X_test, y_train, y_test = split_data(df, target_col)

    #Preprocessor
    preprocessor = build_preprocessor(X_train)

    #Model
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    #Pipeline
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "churn_model_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print("\n Model trained and saved successfully!")


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw_data" / "customer_churn_dataset-training-master.csv"

if __name__ == "__main__":
    train_model(
        data_path=DATA_PATH,
        target_col="Churn"
    )

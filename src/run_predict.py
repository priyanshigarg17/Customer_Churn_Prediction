# run_predict.py

import pandas as pd
from pathlib import Path
from predict import predict_from_dataframe

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model_pipeline.joblib"
DATA_PATH = BASE_DIR / "data" / "raw_data" / "customer_churn_dataset-training-master.csv"

# Load new data (NO Churn column ideally)
df = pd.read_csv(DATA_PATH)

preds, probs = predict_from_dataframe(df, MODEL_PATH)

df["Churn_Prediction"] = preds
df["Churn_Probability"] = probs

print(df.head())

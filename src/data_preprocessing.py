# data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def split_data(df, target_col, test_size=0.2, random_state=42):
    """
    Split dataframe into train-test sets
    """
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
    )


def build_preprocessor(X):
    """
    Create preprocessing pipeline
    """
    # ❌ ID-like columns remove
    drop_cols = [c for c in ["CustomerID", "customer_id", "ID"] if c in X.columns]
    X = X.drop(columns=drop_cols)
    
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category"]).columns

    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor

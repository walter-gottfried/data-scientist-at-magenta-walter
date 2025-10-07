import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

import logging
import sys
from dagster import AssetOut, multi_asset, asset, get_dagster_logger


log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "training"

# ------------------------------------------
# Split train/test
# ------------------------------------------
@multi_asset(
    group_name=group_name,
    outs={
        "train_data": AssetOut(),
        "test_data": AssetOut(),
    },
)
def split_train_test(df_input_preprocessed: pd.DataFrame):
    """
    Splits the preprocessed dataframe into train and test sets.
    Assumes the target column is `has_done_upselling`.
    """
    logger.info("Splitting dataset into train and test sets...")

    target_col = "has_done_upselling"
    X = df_input_preprocessed.drop(columns=[target_col])
    y = df_input_preprocessed[target_col]

    train_X, test_X, train_y, test_y = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_data = pd.concat([train_X, train_y], axis=1)
    test_data = pd.concat([test_X, test_y], axis=1)

    logger.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    logger.info(f"Positive class ratio (train): {train_y.mean():.3f}")

    return train_data, test_data


# ------------------------------------------
# Model training
# ------------------------------------------
@asset(group_name=group_name)
def classifier(train_data: pd.DataFrame):
    """
    Trains a model to predict has_done_upselling using XGBoost + SMOTE.
    Returns a fitted pipeline.
    """
    logger.info("Starting model training...")

    target_col = "has_done_upselling"
    X = train_data.drop(columns=[target_col])
    y = train_data[target_col]

    # Identify categorical and numeric features
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    logger.info(f"Categorical features: {categorical_cols}")
    logger.info(f"Numeric features: {numeric_cols}")

    # Preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # XGBoost model
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=(1 / 0.07),  # helps balance the 7% positive class
        eval_metric="auc",
        n_jobs=-1,
    )

    # Combine SMOTE + preprocessing + model in one pipeline
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    logger.info("Model training completed successfully.")
    return pipeline


# ------------------------------------------
# Evaluation asset
# ------------------------------------------
@asset(group_name=group_name)
def evaluate_classifier(classifier, test_data: pd.DataFrame):
    """
    Evaluates the trained classifier on the test set and logs metrics.
    """
    logger.info("Evaluating classifier...")

    target_col = "has_done_upselling"
    X_test = test_data.drop(columns=[target_col])
    y_test = test_data[target_col]

    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    logger.info(f"AUC: {auc:.4f}")
    logger.info(f"F1: {f1:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred))

    return {
        "auc": auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

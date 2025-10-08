import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import os
import joblib
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score,
    make_scorer
)
#from imblearn.over_sampling import SMOTE
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
    Trains an XGBoost model with feature selection.
    Performs stratified K-fold cross-validation with both train and validation metrics logged.
    """
    logger.info("Starting model training with cross-validation and feature selection...")

    target_col = "has_done_upselling"
    id_cols = ["customer_id", "rating_account_id"]


    X = train_data.drop(columns=[target_col] + id_cols)
    y = train_data[target_col]

    # Identify categorical and numeric features
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    logger.info(f"Categorical features: {categorical_cols}")
    logger.info(f"Numeric features: {numeric_cols}")

    # ------------------------------------------
    # Preprocessing
    # ------------------------------------------
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # ------------------------------------------
    # Base model
    # ------------------------------------------
    base_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=(1 / 0.07),
        eval_metric="auc",
        n_jobs=-1,
    )

    # ------------------------------------------
    # Feature selection
    # ------------------------------------------
    feature_selector = SelectFromModel(
        estimator=XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
        ),
        threshold="median",
        prefit=False
    )

    # ------------------------------------------
    # Full pipeline
    # ------------------------------------------
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            #("smote", SMOTE(random_state=42)),
            ("feature_selection", feature_selector),
            ("model", base_model),
        ]
    )

    # ------------------------------------------
    # Cross-validation (with train + validation metrics)
    # ------------------------------------------
    logger.info("Running stratified 5-fold cross-validation...")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
        "accuracy": "accuracy",
        "f1": "f1",
    }

    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=True,
    )

    # Compute mean/std for each metric (train + validation)
    def summarize_metric(metric_name):
        train_mean = np.mean(cv_results[f"train_{metric_name}"])
        train_std = np.std(cv_results[f"train_{metric_name}"])
        val_mean = np.mean(cv_results[f"test_{metric_name}"])
        val_std = np.std(cv_results[f"test_{metric_name}"])
        return train_mean, train_std, val_mean, val_std

    logger.info("----- Cross-Validation Results (Train vs Validation) -----")
    for metric in ["roc_auc", "f1", "accuracy"]:
        tr_m, tr_s, va_m, va_s = summarize_metric(metric)
        logger.info(
            f"{metric.upper():<9} | Train: {tr_m:.4f} ± {tr_s:.4f} | "
            f"Val: {va_m:.4f} ± {va_s:.4f} | Δ={tr_m - va_m:+.4f}"
        )
    logger.info("----------------------------------------------------------")

    # ------------------------------------------
    # Final fit on all training data
    # ------------------------------------------
    pipeline.fit(X, y)
    logger.info("Final model trained on full training data (with feature selection).")

    # Log final features used
    preprocessor = pipeline.named_steps["preprocessor"]
    encoded_feature_names = preprocessor.get_feature_names_out()
    feature_selector = pipeline.named_steps["feature_selection"]
    mask = feature_selector.get_support()
    selected_features = encoded_feature_names[mask]

    logger.info(f"Model uses {len(selected_features)} features after selection:")
    for feat in selected_features:
        logger.info(f"  - {feat}")

    model_dir = "src/code_location_interview/code_location_interview/assets/magenta_interview/models"
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"classifier_pipeline_{timestamp}.joblib")

    joblib.dump(pipeline, model_path)
    logger.info(f"Saved full pipeline to: {model_path}") # optional (e.g. for API deployment outside Dagster)

    return pipeline
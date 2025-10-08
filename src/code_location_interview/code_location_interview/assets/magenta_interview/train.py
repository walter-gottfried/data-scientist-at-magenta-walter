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
    # to know if ther is an imbalance problem

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
    numeric_transformer = StandardScaler() #technically not necessary for tree-based models
    categorical_transformer = OneHotEncoder(handle_unknown="ignore") 
    #xgboost can handle categorical features natively, but we use 
    #one-hot encoding here for possibility of using shap later (shap can't handle categorical features directly)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # ------------------------------------------
    # Base model
    # ------------------------------------------
    # logistic regression would be possible, but less powerful since the data is likely non-linear
    # use xgboost since it's powerful, fast, can capture non-linear data relationships and handles missing values natively
    # not natively interpretable but can be made interpretable with shap
    # good for tabular data, which is the case here
    # other option could be lightgbm or more advanced models like neural networks
    # limitations: can overfit if not careful, less interpretable than simpler models

    #in real world scenario, multiple models should be compared (e.g. logistic regression, random forest, lightgbm, neural networks)
    # and ideally tracked with mlflow or similar tool for better reproducibility and model management
    # and to make automated model replacement after retraining easier while keeping track of model versions and performance

    # use scale_pos_weight to handle class imbalance, leads to inflated probability estimates, but not a problem for ranking
    base_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        scale_pos_weight=(1 / 0.07),
        eval_metric="auc",
        #auc is a good metric for imbalanced classification compared to accuracy, other option: f1
        n_jobs=-1,
    )

    # set other hyperparameters as default for now, 
    # could (and should) be tuned with grid search or bayesian optimization (e.g. hyperopt) in future


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
        #quick win to cut off all none-informative features, but can be tuned in future to be more precise
        # feature reduction is important to reduce overfitting, improve interpretability and reduce inference time
        prefit=False
    )

    # ------------------------------------------
    # Full pipeline
    # ------------------------------------------
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            #("smote", SMOTE(random_state=42)), #smote can't handle null values, so we don't use it here, but if nulls where imputed in preprocessing, it could be used to handle class imbalance
            ("feature_selection", feature_selector),
            ("model", base_model),
        ]
    )

    # ------------------------------------------
    # Cross-validation (with train + validation metrics)
    # ------------------------------------------
    logger.info("Running stratified 5-fold cross-validation...")

    # in this use case StratifiedGroupKFold would be better to avoid data leakage between customers, 
    # but since the data is randomly sampled, I assume it's not a big issue here
    # in real world cases, not using StratifiedGroupKFold could lead to overly optimistic results
    # since the model could learn customer-specific patterns that won't generalize to new customers
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": make_scorer(roc_auc_score),
        "accuracy": "accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall"
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
    # train + validation metrics are important to detect overfitting (if train >> val, the model is overfitting)
    def summarize_metric(metric_name):
        train_mean = np.mean(cv_results[f"train_{metric_name}"])
        train_std = np.std(cv_results[f"train_{metric_name}"])
        val_mean = np.mean(cv_results[f"test_{metric_name}"])
        val_std = np.std(cv_results[f"test_{metric_name}"])
        return train_mean, train_std, val_mean, val_std

    logger.info("----- Cross-Validation Results (Train vs Validation) -----")
    for metric in ["roc_auc", "f1", "accuracy", "precision", "recall"]:
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

    # the final model turned out to be overfitting and not very well performing, 
    # but as described in the task description this was expected since the data is randomly sampled

    #ROC_AUC   | Train: 0.7876 ± 0.0081 | Val: 0.5629 ± 0.0061 | Δ=+0.2247
    #F1        | Train: 0.2920 ± 0.0087 | Val: 0.1543 ± 0.0036 | Δ=+0.1377
    #ACCURACY  | Train: 0.7015 ± 0.0116 | Val: 0.6457 ± 0.0067 | Δ=+0.0558
    #PRECISION | Train: 0.1748 ± 0.0060 | Val: 0.0925 ± 0.0021 | Δ=+0.0823
    #RECALL    | Train: 0.8876 ± 0.0043 | Val: 0.4667 ± 0.0165 | Δ=+0.4209

    #recall is very high, but precision is very low, so the model predicts almost all customers to do upselling
    # this is likely due to the high class imbalance and the use of scale_pos_weight,
    # which leads to inflated probability estimates
    # this parameter could be tuned in future to find a better balance between precision and recall

    return pipeline
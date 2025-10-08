import logging
import shap
import sys
import pandas as pd
from dagster import get_dagster_logger, asset


log_fmt = "[%(asctime)s] %(message)s"
log_datefmt = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(stream=sys.stdout, format=log_fmt, datefmt=log_datefmt, level=logging.INFO)
logger = get_dagster_logger(__name__)

group_name = "predict"


@asset(
     group_name=group_name,
     
 )
def predictions(classifier, df_input_preprocessed):
    """
    Uses the trained classifier pipeline to make predictions on test data.
    """
    logger.info("Running predictions on test data...")

    # Drop ID columns
    X_test = df_input_preprocessed.drop(columns=["has_done_upselling", "customer_id", "rating_account_id"], errors="ignore")
    y_true = df_input_preprocessed["has_done_upselling"]

    # Predict probabilities
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Compute SHAP values
    logger.info("Computing SHAP values for top features...")

     # Transform input using pipeline steps
    preprocessed_X = classifier.named_steps["preprocessor"].transform(X_test)
    selected_X = classifier.named_steps["feature_selection"].transform(preprocessed_X)

    # Feature names after preprocessing & selection
    encoded_features = classifier.named_steps["preprocessor"].get_feature_names_out()
    mask = classifier.named_steps["feature_selection"].get_support()
    selected_features = encoded_features[mask]

    # Use TreeExplainer for XGBoost pipeline
    model = classifier.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(selected_X)

    shap_df = pd.DataFrame(shap_values.values, columns=selected_features, index=X_test.index)

    # ------------------------------
    # Top 3 features per prediction
    # ------------------------------
    top_features = []
    for i, row in shap_df.iterrows():
        top3 = row.abs().sort_values(ascending=False).head(3).index.tolist()
        top_features.append(top3)


    # Build result dataframe
    predictions_df = pd.DataFrame({
        "customer_id": df_input_preprocessed.get("customer_id", pd.Series(range(len(y_true)))),
        "true_label": y_true,
        "predicted_label": y_pred,
        "predicted_proba": y_pred_proba,
        "top_3_features": top_features
    })

    logger.info(f"Generated {len(predictions_df)} predictions.")
    logger.info(f"Mean predicted probability: {y_pred_proba.mean():.4f}")
    return predictions_df

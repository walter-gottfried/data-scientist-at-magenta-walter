import pandas as pd 
from dagster import asset
def summarize_customer_interactions(customer_interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a pivot table summarizing customer interactions by type/subtype,
    including total interactions and minimum days since last interaction.

    Parameters
    ----------
    customer_interactions : pandas.DataFrame
        Must contain columns:
        - 'customer_id'
        - 'type_subtype'
        - 'n'
        - 'days_since_last'

    Returns
    -------
    pandas.DataFrame
        Pivot table with one row per customer, columns for each interaction type/subtype,
        and additional summary columns:
        - 'min_days_since_last'
        - 'sum_n'
    """
    # Create pivot table
    customer_interactions_pivot = customer_interactions.pivot_table(
        values=['n', 'days_since_last'],
        index='customer_id',
        columns='type_subtype',
        observed=True
    )

    customer_interactions_pivot['n'] = customer_interactions_pivot['n'].fillna(0)

    # Add summary columns
    customer_interactions_pivot['min_days_since_last'] = customer_interactions_pivot['days_since_last'].min(axis=1)
    customer_interactions_pivot['sum_n'] = customer_interactions_pivot['n'].sum(axis=1)

    return customer_interactions_pivot

def add_remaining_binding_flag(core_data: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'has_remaining_binding_days' column to the DataFrame,
    where 1 indicates remaining_binding_days > 0, and 0 otherwise.
    """
    if "remaining_binding_days" not in core_data.columns:
        raise KeyError("DataFrame must contain 'remaining_binding_days' column.")
    
    core_data["has_remaining_binding_days"] = (
        core_data["remaining_binding_days"] > 0
    ).astype(int)
    
    return core_data


def aggregate_usage_info(core_data: pd.DataFrame, usage_info: pd.DataFrame) -> pd.DataFrame:
    """
    Merges core_data with usage_info, computes leftover data, 
    and aggregates usage statistics by 'rating_account_id'.

    Parameters:
        core_data (pd.DataFrame): Core customer data containing 'rating_account_id' and 'available_gb'.
        usage_info (pd.DataFrame): Usage information containing 'rating_account_id' and 'used_gb'.

    Returns:
        pd.DataFrame: Aggregated statistics (mean, std, min, max for 'leftover_data_gb' and 'used_gb', 
                      sum for 'has_used_roaming') grouped by 'rating_account_id'.
    """

    # --- Merge core data with usage info ---
    core_data_usage_info_merged = core_data.merge(
        usage_info,
        on="rating_account_id",
        how="left"
    )

    # --- Compute leftover data ---
    core_data_usage_info_merged["leftover_data_gb"] = (
        core_data_usage_info_merged["available_gb"] - core_data_usage_info_merged["used_gb"]
    )

    # --- Aggregate usage information ---
    aggregated_usage_info = core_data_usage_info_merged.groupby("rating_account_id").agg(
        {
            "leftover_data_gb": ["mean", "std", "min", "max"],
            "used_gb": ["mean", "std", "min", "max"],
            "has_used_roaming": "sum",
        }
    )

    return aggregated_usage_info


def prepare_merged_data(
    core_data: pd.DataFrame,
    customer_interactions_pivot: pd.DataFrame,
    aggregated_usage_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Flattens multi-level columns, merges core data with customer interactions 
    and aggregated usage info, and fills missing values, without mutating
    the input DataFrames.

    Parameters:
        core_data (pd.DataFrame): Base customer dataset (must include 'customer_id' and 'rating_account_id').
        customer_interactions_pivot (pd.DataFrame): Pivot table of customer interactions.
        aggregated_usage_info (pd.DataFrame): Aggregated usage information grouped by rating account.

    Returns:
        pd.DataFrame: Cleaned and merged dataset ready for analysis or modeling.
    """

    # --- Create copies to avoid modifying original data ---
    # could be solved more elegantly by flattening columns in the function that creates them but it works for now
    core_copy = core_data.copy(deep=True)
    interactions_copy = customer_interactions_pivot.copy(deep=True)
    usage_copy = aggregated_usage_info.copy(deep=True)

    # --- Flatten multi-level columns safely ---
    interactions_copy.columns = [
        "_".join([str(c) for c in col if c]) if isinstance(col, tuple) else str(col)
        for col in interactions_copy.columns
    ]

    usage_copy.columns = [
        "_".join([str(c) for c in col if c]) if isinstance(col, tuple) else str(col)
        for col in usage_copy.columns
    ]

    # --- Merge datasets ---
    merged_data = (
        core_copy
        .merge(interactions_copy, on="customer_id", how="left")
        .merge(usage_copy, on="rating_account_id", how="left")
    )

    # --- Fill missing values ---
    if "sum_n" in merged_data.columns:
        merged_data["sum_n"] = merged_data["sum_n"].fillna(0)

    merged_data = merged_data.fillna({
        col: 0 for col in merged_data.columns if col.startswith("n_")
    })

    return merged_data

def transform_data(
    core_data: pd.DataFrame,
    customer_interactions: pd.DataFrame,
    usage_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Transforms and merges core data, customer interactions, and usage info into a single DataFrame.

    Parameters:
        core_data (pd.DataFrame): Core customer data.
        customer_interactions (pd.DataFrame): Customer interaction records.
        usage_info (pd.DataFrame): Usage information records.

    Returns:
        pd.DataFrame: Transformed and merged dataset ready for analysis or modeling.
    """
    # Summarize customer interactions
    customer_interactions_pivot = summarize_customer_interactions(customer_interactions)

    # Add remaining binding days flag
    core_data_with_flag = add_remaining_binding_flag(core_data)

    # Aggregate usage information
    aggregated_usage_info = aggregate_usage_info(core_data_with_flag, usage_info)

    # Prepare merged data
    merged_data = prepare_merged_data(
        core_data_with_flag,
        customer_interactions_pivot,
        aggregated_usage_info
    )

    return merged_data  

# ------------------------------
# Dagster asset
# ------------------------------
@asset
def df_input_preprocessed(
    core_data: pd.DataFrame,
    customer_interactions: pd.DataFrame,
    usage_info: pd.DataFrame
) -> pd.DataFrame:
    """
    Dagster asset that produces the fully merged and transformed dataset.
    """
    return transform_data(core_data, customer_interactions, usage_info)




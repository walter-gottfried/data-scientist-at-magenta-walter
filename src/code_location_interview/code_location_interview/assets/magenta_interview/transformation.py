def summarize_customer_interactions(customer_interactions):
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
import pandas as pd
from code_location_interview.assets.magenta_interview.transformation import summarize_customer_interactions

def test_summarize_customer_interactions_real_types():
    # Sample data
    data = {
        'customer_id': [1, 1, 2, 2, 3, 3],
        'type_subtype': [
            'rechnungsanfragen',
            'produkte&services-tarifdetails',
            'prolongation',
            'produkte&services-tarifwechsel',
            'rechnungsanfragen',
            'prolongation'
        ],
        'n': [2, 1, 3, 2, 4, 1],
        'days_since_last': [5, 10, 7, 20, 3, 15]
    }
    df = pd.DataFrame(data)

    # Run the function
    pivot = summarize_customer_interactions(df)

    # Check that all types exist in the pivot columns
    expected_types = [
        'rechnungsanfragen',
        'produkte&services-tarifdetails',
        'prolongation',
        'produkte&services-tarifwechsel'
    ]
    for col_type in expected_types:
        assert ('n', col_type) in pivot.columns, f"Missing n column for {col_type}"
        assert ('days_since_last', col_type) in pivot.columns, f"Missing days_since_last column for {col_type}"

    # Test summary columns
    assert 'min_days_since_last' in pivot.columns, "Missing 'min_days_since_last'"
    assert 'sum_n' in pivot.columns, "Missing 'sum_n'"

    # Check calculations for customer 1
    # n total = 2+1=3, min_days_since_last = min(5,10)=5
    assert pivot.loc[1, 'sum_n'].item() == 3, "Incorrect sum_n for customer 1"
    assert pivot.loc[1, 'min_days_since_last'].item() == 5, "Incorrect min_days_since_last for customer 1"

    # Check calculations for customer 3
    # n total = 4+1=5, min_days_since_last = min(3,15)=3
    assert pivot.loc[3, 'sum_n'].item() == 5, "Incorrect sum_n for customer 3"
    assert pivot.loc[3, 'min_days_since_last'].item() == 3, "Incorrect min_days_since_last for customer 3"
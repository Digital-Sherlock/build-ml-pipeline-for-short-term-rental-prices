'''
This module performs multiple deterministic
and non-deterministic tests against supplied
dataset(s). For non-deterministic tests the A|B
testing approach was followed with two datasets -
sample and reference.
'''


import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):
    '''
    Deterministic test that checks whether
    supplied dataset has the desired columns.

    Input:
        - data: (pd.DataFrame) supplied to pytest via CLI;
                fixture is in conftest.py
    Output:
        - None
    '''

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):
    '''
    Deterministic test that checks for
    neighborhoods in the supplied dataset.

    Input:
        - data: (pd.DataFrame) supplied to pytest via CLI;
                fixture is in conftest.py
    Output:
        - None
    '''

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    """
    Test proper longitude and latitude boundaries for properties in and around NYC.

    Input:
        - data: (pd.DataFrame) supplied to pytest via CLI;
                fixture is in conftest.py
    Output:
        - None
    """
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    """
    Apply a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset.

    Input:
        - data: (pd.DataFrame) sample dataset
        - ref_data: (pd.DataFrame) reference dataset
        - kl_threshold: (float) kl_threshold
    Output:
        - None

    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold


def test_row_count(data):
    '''
    Test row count is within the desired range.

    Input:
        - data: (pd.DataFrame) sample dataset
    Output:
        - None
    '''
    assert 15000 < data.shape[0] < 1000000


def test_price_range(data, min_price, max_price):
    '''
    Test the price range.

    Input:
        - data: (pd.DataFrame) sample dataset
        - min_price: (float) min price
        - max_price: (float) max price
    Output:
        - None
    '''
    assert data['price'].between(min_price, max_price).unique()[0]
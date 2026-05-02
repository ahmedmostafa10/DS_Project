import os

import pandas as pd
import numpy as np
import pytest

from scripts.data.data_cleaning import remove_irrelevant_columns,accuracy_rule_based_correction,\
      accuracy_qurantine_based_fixing, high_missingness_removal,fill_district_with_mode,\
      impute_missing, drop_duplicates,apply_clipping, handle_outliers_with_clipping,\
      DISTANCE_COLUMNS, COUNT_COLUMNS

def test_remove_irrelevant_columns():
    df = pd.DataFrame({
        "listing_id": [1, 2],
        "price": [100, 200],
        "area_value": [50, 60]
    })

    result = remove_irrelevant_columns(df, ["listing_id"])

    assert "listing_id" not in result.columns
    assert "price" in result.columns
    assert "area_value" in result.columns


def test_accuracy_rule_based_correction():
    df = pd.DataFrame({
        "bedrooms": ["studio", "2", "3"]
    })

    result = accuracy_rule_based_correction(df)

    assert result["bedrooms"].tolist() == [1, 2, 3]
    assert result["bedrooms"].dtype == int


def test_accuracy_quarantine():
    quarantine_log_path = "test_quarantine_log.csv"

    df = pd.DataFrame({
        "area_value": [50, 2000],
        "price_egp": [100000, 50000000],
        "lon": [30, 60],
        "lat": [25, 10],
        "bedrooms": [2, 0]
    })

    result = accuracy_qurantine_based_fixing(df, quarantine_log_path=quarantine_log_path)

    # only valid rows remain
    assert len(result) == 1

    # quarantine file created
    assert os.path.exists(quarantine_log_path)


def test_high_missingness_removal():
    df = pd.DataFrame({
        "is_exclusive": [None, None],
        "amenities": [None, None],
        "keep": [1, 2]
    })

    result = high_missingness_removal(df)

    assert "is_exclusive" not in result.columns
    assert "amenities" not in result.columns
    assert "keep" in result.columns

def test_fill_district_with_mode():
    df = pd.DataFrame({
        "city": ["A", "A", "A"],
        "town": ["T1", "T1", "T1"],
        "district": ["D1", np.nan, "D1"]
    })

    df["district"] = df["district"].astype("category")

    result, missing = fill_district_with_mode(df)

    assert missing == 1
    assert result["district"].isna().sum() == 0
    assert np.unique(result["district"]).size == 1

def test_impute_missing():
    df = pd.DataFrame({
        "city": ["A"],
        "town": ["T"],
        "district": pd.Series([np.nan], dtype="category"),
        "furnished": pd.Series([np.nan], dtype="category"),
        "completion_status": ["done"],
        "bathroom": [1]
    })

    result = impute_missing(df)

    assert "Missing" in result["furnished"].values 
    assert result["furnished"].isna().sum() == 0
    assert result["district"].isna().sum() == 0

def test_drop_duplicates():
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": [3, 3, 4]
    })

    result = drop_duplicates(df)

    assert len(result) == 2

def test_apply_clipping():
    df = pd.DataFrame({
        "x": [1, 2, 3, 100]
    })

    result, count = apply_clipping(df, "x", 0.0, 0.75)

    assert count >= 0
    assert result["x"].max() <= df["x"].quantile(0.75)


def test_handle_outliers():
    df = pd.DataFrame({
        "area_value": [10, 20, 10000],
        "dist_nearest_school_km": [1, 2, 100],
        "school_count_within_3km": [1, 2, 100]
    })

    result = handle_outliers_with_clipping(df, distance_columns=["dist_nearest_school_km"], \
                                           count_columns=["school_count_within_3km"])

    assert result.shape[0] == 3
    assert "area_value" in result.columns
import os

import pandas as pd
import numpy as np
import pytest

from scripts.data.data_cleaning import fix_consistency, remove_irrelevant_columns,accuracy_rule_based_correction,\
      accuracy_qurantine_based_fixing, high_missingness_removal,fill_district_with_mode,\
      fix_missingness, drop_duplicates,apply_clipping, handle_outliers_with_clipping,\
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

def test_fix_consistency():
    df = pd.DataFrame({
        "property_type":    ["Apartments", "apartment", "apartment", "apartment", "apartment", "apartment"],
        "offering_type":    ["for-sale", "for-sale", "for-sale", "for-sale", "for-sale", "for-sale"],
        "completion_status":["completed", "off_plan", "completed_primary", "off_plan_primary", "completed", "completed"],
        "town":             ["Nasr City", "Maadi", "Maadi", "Maadi", "Maadi", "Maadi"],
        "district":         ["D1", "D2", "  6th October!! ", "d1", "d1", "d1"],
        "furnished":        ["YES", "NO", "PARTLY", "YES", "NO", "YES"],
        "price_period":     ["monthly"] * 6,
        "price_currency":   ["EGP"] * 6,
        "area_unit":        ["sqm"] * 6,
        "is_verified":      [True] * 6,
        "is_new_construction": [False] * 6,
        "rera":             [None] * 6,
    })

    result = fix_consistency(df)

    # row count unchanged
    assert len(result) == len(df)

    # property_type unified
    if "property_type" in result.columns:
        assert (result["property_type"] == "apartment").all()

    # completion_status normalized
    assert "completed_primary" not in result["completion_status"].values
    assert "off_plan_primary" not in result["completion_status"].values

    # town: City word removed
    assert all("City" not in t for t in result["town"].values)

    # district: lowercased, no punctuation
    assert all(d == d.lower() for d in result["district"].values)
    assert all("!" not in d for d in result["district"].values)

    # furnished mapped
    assert "YES" not in result["furnished"].values
    assert "NO" not in result["furnished"].values
    assert "PARTLY" not in result["furnished"].values

    # single-value columns dropped
    for col in ["price_period", "price_currency", "area_unit", "is_verified", "is_new_construction", "rera"]:
        assert col not in result.columns

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

    result = fix_missingness(df)

    assert "Missing" in result["furnished"].values 
    assert result["furnished"].isna().sum() == 0
    assert result["district"].isna().sum() == 0

    df_2 = pd.DataFrame({
        "city": ["A", "A"],
        "town": ["T", "T"],
        "district": pd.Series(["6th october", "6th october"], dtype="category"),
        "furnished": pd.Series(["yes", "yes"], dtype="category"),
        "completion_status": ["done", np.nan],
        "bathroom": [np.nan,1]
    })

    result_2 = fix_missingness(df_2)
    assert result_2.shape[0] == 0


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
        "dist_nearest_transit_station_km": [1, 2, 100],
        "school_count_within_3km": [1, 2, 100]
    })

    result = handle_outliers_with_clipping(df, clipping_columns=["dist_nearest_transit_station_km"], \
                                           log_columns=["school_count_within_3km"])

    assert result.shape[0] == 3
    assert "area_value" in result.columns
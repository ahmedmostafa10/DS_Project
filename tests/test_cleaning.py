import os

import numpy as np
import pandas as pd

from src.data.data_cleaning import (
    accuracy_qurantine_based_fixing,
    accuracy_rule_based_correction,
    apply_clipping,
    drop_duplicates,
    fill_district_with_mode,
    fix_consistency,
    fix_missingness,
    handle_outliers_with_clipping,
    high_missingness_removal,
    remove_irrelevant_columns,
)


def test_remove_irrelevant_columns():
    df = pd.DataFrame({"listing_id": [1, 2], "price": [100, 200], "area_value": [50, 60]})

    result = remove_irrelevant_columns(df, ["listing_id"])

    assert "listing_id" not in result.columns
    assert "price" in result.columns
    assert "area_value" in result.columns


def test_accuracy_rule_based_correction():
    df = pd.DataFrame({"bedrooms": ["studio", "2", "3"]})

    result = accuracy_rule_based_correction(df)

    assert result["bedrooms"].tolist() == [1, 2, 3]
    assert result["bedrooms"].dtype == int


def test_accuracy_quarantine():
    quarantine_log_path = "test_quarantine_log.csv"

    df = pd.DataFrame(
        {
            "area_value": [50, 2000],
            "price_egp": [100000, 50000000],
            "lon": [30, 60],
            "lat": [25, 10],
            "bedrooms": [2, 0],
        }
    )

    result = accuracy_qurantine_based_fixing(df, quarantine_log_path=quarantine_log_path)

    # only valid rows remain
    assert len(result) == 1

    # quarantine file created
    assert os.path.exists(quarantine_log_path)


def test_fix_consistency():
    df = pd.DataFrame(
        {
            "property_type": [
                "Apartments",
                "apartment",
                "apartment",
                "apartment",
                "apartment",
                "apartment",
            ],
            "offering_type": [
                "for-sale",
                "for-sale",
                "for-sale",
                "for-sale",
                "for-sale",
                "for-sale",
            ],
            "completion_status": [
                "completed",
                "off_plan",
                "completed_primary",
                "off_plan_primary",
                "completed",
                "completed",
            ],
            "town": [
                "Nasr City",
                "Maadi",
                "Maadi",
                "Maadi",
                "Maadi",
                "Maadi",
            ],
            "district": [
                "D1",
                "D2",
                "  6th October!! ",
                "d1",
                "d1",
                "d1",
            ],
            "furnished": [
                "YES",
                "NO",
                "PARTLY",
                "YES",
                "NO",
                "YES",
            ],
            "price_period": ["monthly"] * 6,
            "price_currency": ["EGP"] * 6,
            "area_unit": ["sqm"] * 6,
            "is_verified": [True] * 6,
            "is_new_construction": [False] * 6,
            "rera": [None] * 6,
            # Required numerical columns
            "price_egp": [1000000, 2000000, 3000000, 4000000, 5000000, 6000000],
            "area": [100, 120, 140, 160, 180, 200],
            # Required bedroom/bathroom columns
            "bedrooms": ["1", "2", "3", "4", "5", "6"],
            "bathroom": ["1", "2", "3", "4", "7+", "none"],
            # Required boolean columns
            "has_parking": [True, False, True, False, True, False],
            # Required coordinate columns
            "lat": [
                30.123456789,
                30.123456789,
                30.123456789,
                30.123456789,
                30.123456789,
                30.123456789,
            ],
            "lon": [
                31.987654321,
                31.987654321,
                31.987654321,
                31.987654321,
                31.987654321,
                31.987654321,
            ],
            "area_value": [100, 120, 140, 160, 180, 200],
            "dist_nearest_school_km": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "school_count_within_3km": [1, 2, 3, 4, 5, 6],
            "dist_nearest_hospital_km": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "hospital_count_within_3km": [1, 2, 3, 4, 5, 6],
            "dist_nearest_supermarket_km": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "supermarket_count_within_3km": [1, 2, 3, 4, 5, 6],
            "dist_nearest_mall_km": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "mall_count_within_3km": [1, 2, 3, 4, 5, 6],
            "dist_nearest_transit_station_km": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "transit_station_count_within_3km": [1, 2, 3, 4, 5, 6],
            "dist_nearest_cafe_restaurant_km": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "cafe_restaurant_count_within_3km": [1, 2, 3, 4, 5, 6],
            "is_premium": [0, 1, 0, 1, 0, 1],
            "is_featured": [1, 0, 1, 0, 1, 0],
            "is_exclusive": [True, False, True, False, True, False],
            "amenities": [
                "Pool, Gym",
                "Parking",
                "Pool",
                "Gym",
                "Parking, Pool",
                "Gym, Parking",
            ],
            "listing_level": [
                "standard",
                "featured",
                "premium",
                "hot",
                "superhot",
                "standard",
            ],
            "city": ["Cairo", "Giza", "Alex", "Cairo", "Giza", "Alex"],
        }
    )

    result = fix_consistency(df)

    # row count unchanged
    assert len(result) == len(df)

    # property_type unified then dropped (single value)
    assert "property_type" not in result.columns

    # completion_status normalized
    assert "completed_primary" not in result["completion_status"].values
    assert "off_plan_primary" not in result["completion_status"].values

    # town cleaned
    assert all("City" not in t for t in result["town"].astype(str).values)

    # district normalized
    assert all(d == d.lower() for d in result["district"].astype(str).values)
    assert all("!" not in d for d in result["district"].astype(str).values)

    # furnished mapped
    assert "YES" not in result["furnished"].values
    assert "NO" not in result["furnished"].values
    assert "PARTLY" not in result["furnished"].values

    assert set(result["furnished"].unique()) == {
        "furnished",
        "unfurnished",
        "partly",
    }

    # bathroom standardized
    assert 7 in result["bathroom"].values
    assert result["bathroom"].isna().sum() >= 1

    # coordinates rounded
    assert result["lat"].iloc[0] == round(30.123456789, 6)
    assert result["lon"].iloc[0] == round(31.987654321, 6)

    # single-value columns dropped
    for col in [
        "price_period",
        "price_currency",
        "area_unit",
        "is_verified",
        "is_new_construction",
        "rera",
        "offering_type",
    ]:
        assert col not in result.columns


def test_high_missingness_removal():
    df = pd.DataFrame({"is_exclusive": [None, None], "amenities": [None, None], "keep": [1, 2]})

    result = high_missingness_removal(df)

    assert "is_exclusive" not in result.columns
    assert "amenities" not in result.columns
    assert "keep" in result.columns


def test_fill_district_with_mode():
    df = pd.DataFrame(
        {
            "city": ["A", "A", "A"],
            "town": ["T1", "T1", "T1"],
            "district": ["D1", np.nan, "D1"],
        }
    )

    df["district"] = df["district"].astype("category")

    result, missing = fill_district_with_mode(df)

    assert missing == 1
    assert result["district"].isna().sum() == 0
    assert np.unique(result["district"]).size == 1


def test_impute_missing():
    df = pd.DataFrame(
        {
            "city": ["A"],
            "town": ["T"],
            "district": pd.Series([np.nan], dtype="category"),
            "furnished": pd.Series([np.nan], dtype="category"),
            "completion_status": ["done"],
            "bathroom": [1],
        }
    )

    result = fix_missingness(df)

    assert "Missing" in result["furnished"].values
    assert result["furnished"].isna().sum() == 0
    assert result["district"].isna().sum() == 0

    df_2 = pd.DataFrame(
        {
            "city": ["A", "A"],
            "town": ["T", "T"],
            "district": pd.Series(["6th october", "6th october"], dtype="category"),
            "furnished": pd.Series(["yes", "yes"], dtype="category"),
            "completion_status": ["done", np.nan],
            "bathroom": [np.nan, 1],
        }
    )

    result_2 = fix_missingness(df_2)
    assert result_2.shape[0] == 0


def test_drop_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})

    result = drop_duplicates(df)

    assert len(result) == 2


def test_apply_clipping():
    df = pd.DataFrame({"x": [1, 2, 3, 100]})

    result, count = apply_clipping(df, "x", 0.0, 0.75)

    assert count >= 0
    assert result["x"].max() <= df["x"].quantile(0.75)


def test_handle_outliers():
    df = pd.DataFrame(
        {
            "area_value": [10, 20, 10000],
            "dist_nearest_transit_station_km": [1, 2, 100],
            "school_count_within_3km": [1, 2, 100],
        }
    )

    result = handle_outliers_with_clipping(
        df,
        clipping_columns=["dist_nearest_transit_station_km"],
        log_columns=["school_count_within_3km"],
    )

    assert result.shape[0] == 3
    assert "area_value" in result.columns

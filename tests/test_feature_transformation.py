import numpy as np
import pandas as pd
import pytest

from src.features.feature_transformation import FeatureTransformation


def make_df(n=60, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "price_egp": rng.uniform(500_000, 5_000_000, n),
            "lat": rng.uniform(25.0, 31.0, n),
            "lon": rng.uniform(28.0, 35.0, n),
            "area_value": rng.uniform(60, 400, n),
            "bedrooms": rng.integers(1, 6, n).astype(float),
            "bathroom": rng.integers(1, 4, n).astype(float),
            "dist_nearest_school_km": rng.uniform(0.1, 5.0, n),
            "dist_nearest_hospital_km": rng.uniform(0.1, 8.0, n),
            "dist_nearest_supermarket_km": rng.uniform(0.1, 3.0, n),
            "dist_nearest_mall_km": rng.uniform(0.5, 10.0, n),
            "dist_nearest_transit_station_km": rng.uniform(0.1, 4.0, n),
            "dist_nearest_cafe_restaurant_km": rng.uniform(0.1, 2.0, n),
            "school_count_within_3km": rng.integers(0, 10, n).astype(float),
            "hospital_count_within_3km": rng.integers(0, 5, n).astype(float),
            "supermarket_count_within_3km": rng.integers(0, 8, n).astype(float),
            "mall_count_within_3km": rng.integers(0, 3, n).astype(float),
            "transit_station_count_within_3km": rng.integers(0, 6, n).astype(float),
            "cafe_restaurant_count_within_3km": rng.integers(0, 15, n).astype(float),
            "listing_level": rng.choice(["standard", "featured", "premium", "hot", "superhot"], n),
            "completion_status": rng.choice(["under-construction", "off_plan", "completed"], n),
            "furnished": rng.choice(["Unfurnished", "Unknown", "Furnished", "PARTLY"], n),
            "city": rng.choice(["Cairo", "Giza", "Alex"], n),
            "town": rng.choice(["Nasr City", "Dokki", "Maadi"], n),
            "district": rng.choice(["D1", "D2", "D3", "D4"], n),
            "is_premium": rng.integers(0, 2, n),
            "is_featured": rng.integers(0, 2, n),
            "has_pool": rng.choice([True, False], n),
            "has_gym": rng.choice([True, False], n),
        }
    )


@pytest.fixture
def ft():
    return FeatureTransformation()


@pytest.fixture
def raw_df():
    return make_df()


@pytest.fixture
def df_target(ft, raw_df):
    return ft.create_target(raw_df.copy())


@pytest.fixture
def splits(ft, df_target):
    return ft.split_data(df_target)


@pytest.fixture
def scaled(ft, splits):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    X_train, X_val, X_test = ft.scale_features(X_train.copy(), X_val.copy(), X_test.copy())
    return X_train, X_val, X_test, y_train, y_val, y_test


@pytest.fixture
def encoded(ft, scaled):
    X_train, X_val, X_test, y_train, y_val, y_test = scaled
    X_train, X_val, X_test = ft.encode_features(X_train.copy(), X_val.copy(), X_test.copy())
    return X_train, X_val, X_test, y_train, y_val, y_test


def test_log_adds_entry(ft):
    ft.log_transformation_action("S", "col", "A", "R")
    assert len(ft.transformation_log_report) == 1
    assert ft.transformation_log_report[0]["stage"] == "S"


def test_create_target_adds_bin_column(ft, raw_df):
    result = ft.create_target(raw_df.copy())
    assert "price_egp_bin" in result.columns
    assert result["price_egp_bin"].nunique() == 3
    assert result["price_egp_bin"].isnull().sum() == 0


def test_split_data_shapes(ft, df_target):
    X_train, X_val, X_test, y_train, y_val, y_test = ft.split_data(df_target)
    assert len(X_train) + len(X_val) + len(X_test) == len(df_target)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)

    assert "price_egp" not in X_train.columns
    assert "price_egp_bin" not in X_train.columns


def test_scale_features_normalizes(ft, splits):
    X_train, X_val, X_test, *_ = splits
    X_train_s, _, _ = ft.scale_features(X_train.copy(), X_val.copy(), X_test.copy())
    assert abs(X_train_s["area_value"].mean()) < 0.1
    assert X_train_s.shape == X_train.shape


def test_encode_features(ft, scaled):
    X_train, X_val, X_test, *_ = scaled
    X_train_e, _, _ = ft.encode_features(X_train.copy(), X_val.copy(), X_test.copy())
    assert pd.api.types.is_numeric_dtype(X_train_e["listing_level"])
    assert "completion_status" not in X_train_e.columns
    assert "furnished" not in X_train_e.columns
    assert X_train_e.isnull().sum().sum() == 0


def test_add_arithmetic_features(ft, encoded):
    X_train, *_ = encoded

    out = ft.add_arithmetic_features(X_train.copy())

    assert "area_per_bedroom" in out.columns
    assert "area_per_bathroom" in out.columns
    assert "bathroom_per_bedroom" in out.columns
    assert "total_rooms" in out.columns
    assert "total_services_count_3km" in out.columns

    expected_area_per_bedroom = X_train["area_value"] / X_train["bedrooms"]
    pd.testing.assert_series_equal(
        out["area_per_bedroom"], expected_area_per_bedroom, check_names=False
    )

    expected_area_per_bathroom = X_train["area_value"] / X_train["bathroom"]
    pd.testing.assert_series_equal(
        out["area_per_bathroom"], expected_area_per_bathroom, check_names=False
    )

    expected_bathroom_per_bedroom = X_train["bathroom"] / X_train["bedrooms"]
    pd.testing.assert_series_equal(
        out["bathroom_per_bedroom"], expected_bathroom_per_bedroom, check_names=False
    )

    expected_total_rooms = X_train["bathroom"] + X_train["bedrooms"]
    pd.testing.assert_series_equal(out["total_rooms"], expected_total_rooms, check_names=False)

    count_cols = [
        "school_count_within_3km",
        "hospital_count_within_3km",
        "supermarket_count_within_3km",
        "mall_count_within_3km",
        "transit_station_count_within_3km",
        "cafe_restaurant_count_within_3km",
    ]

    expected_services = X_train[count_cols].sum(axis=1)

    pd.testing.assert_series_equal(
        out["total_services_count_3km"], expected_services, check_names=False
    )


def test_fit_and_apply_location_stats(ft, encoded):
    X_train, *_ = encoded

    X_train = ft.add_arithmetic_features(X_train.copy())

    district_stats, town_stats, global_stats = ft.fit_location_stats(X_train)

    assert isinstance(district_stats, pd.DataFrame)
    assert isinstance(town_stats, pd.DataFrame)
    assert isinstance(global_stats, dict)

    assert "district_avg_area" in district_stats.columns
    assert "town_avg_area" in town_stats.columns
    assert "area_value" in global_stats


def test_apply_location_stats(ft, encoded):
    X_train, X_val, *_ = encoded

    X_train = ft.add_arithmetic_features(X_train.copy())
    X_val = ft.add_arithmetic_features(X_val.copy())

    district_stats, town_stats, global_stats = ft.fit_location_stats(X_train)

    out = ft.apply_location_stats(X_val.copy(), district_stats, town_stats, global_stats)

    assert "district_avg_area" in out.columns
    assert "town_avg_area" in out.columns
    assert "area_vs_district_avg" in out.columns

    assert out["district_avg_area"].isna().sum() == 0
    assert out["town_avg_area"].isna().sum() == 0

    expected_ratio = (out["area_value"] - out["district_avg_area"]) / out["district_avg_area"]

    pd.testing.assert_series_equal(out["area_vs_district_avg"], expected_ratio, check_names=False)
    assert np.isfinite(out["area_vs_district_avg"]).all()


def test_apply_binary_features(ft, encoded):
    X_train, *_ = encoded

    area_median = X_train["area_value"].median()
    out = ft.apply_binary_features(X_train.copy(), area_median)

    assert "is_large_house" in out.columns
    assert "is_small_house" in out.columns
    assert "near_school" in out.columns
    assert "near_mall" in out.columns
    assert "high_quality_listing" in out.columns

    assert (out["is_large_house"] == (X_train["area_value"] > area_median)).all()
    assert (out["is_small_house"] == (X_train["area_value"] < area_median)).all()
    assert (out["near_school"] == (X_train["dist_nearest_school_km"] < 1)).all()
    assert (out["near_mall"] == (X_train["dist_nearest_mall_km"] < 2)).all()


def test_find_correlated_features(ft):
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [1.0, 2.0, 3.0, 4.0, 5.0],
            "C": [5.0, 3.0, 1.0, 2.0, 4.0],
        }
    )
    result = ft.find_correlated_features(df.corr(), threshold=0.9)
    assert isinstance(result, pd.DataFrame)
    pairs = set(zip(result["feature_A"], result["feature_B"]))
    assert ("A", "B") in pairs or ("B", "A") in pairs


def test_save_outputs(ft, encoded):
    X_train, X_val, X_test, y_train, y_val, y_test = encoded
    ft.log_transformation_action("S", "C", "A", "R")
    saved = {}
    with pytest.MonkeyPatch().context() as m:
        m.setattr(
            pd.DataFrame, "to_csv", lambda self, path, **kw: saved.__setitem__(path, self.copy())
        )
        ft.save_outputs(X_train, X_val, X_test, y_train, y_val, y_test)
    assert "target" in saved["./data/processed/train.csv"].columns
    assert len(saved) == 4

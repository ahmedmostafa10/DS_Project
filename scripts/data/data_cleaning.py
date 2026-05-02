import os
import logging

import numpy as np
import pandas as pd


#################       CONSTANTS       #####################

RAW_DATA_PATH= "./../../data/raw/data.csv"
CLEANED_DATA_PATH="./../../data/cleaned/cleaned_data.csv"
QUARANTINE_LOG_PATH="./../../data/cleaned/cleaning_quarantined_data.csv"
CLEANING_LOG_REPORT_PATH="./../../reports/cleaning_log_report.csv"
CLEANING_LOGGING_PATH="./../../reports/cleaning.log"

IRRELEVANT_COLUMNS_TO_REMOVE = [
        "listing_id","internal_id","detail_url","title","images_count","has_video","video_url","reference","description",
        "agent_id","agent_name","agent_email","agent_is_verified","agent_languages","broker_id","broker_name","broker_email",
        "broker_phone","contact_phone","contact_whatsapp","location_full","contact_email","scraped_at","source","id","url","title.1","scraped_at.1",
        "category", "listing_type","listed_date","subdistrict", "payment_plan","is_direct_from_developer"
    ]

DISTANCE_COLUMNS = [
    "dist_nearest_school_km", "dist_nearest_hospital_km",
    "dist_nearest_supermarket_km", "dist_nearest_mall_km",
    "dist_nearest_transit_station_km", "dist_nearest_cafe_restaurant_km",
]

COUNT_COLUMNS = [
    "school_count_within_3km", "hospital_count_within_3km",
    "supermarket_count_within_3km", "mall_count_within_3km",
    "transit_station_count_within_3km", "cafe_restaurant_count_within_3km",
]


CLIPPING_COLUMNS=["dist_nearest_transit_station_km","mall_count_within_3km","transit_station_count_within_3km"]
LOG_COLUMNS=["dist_nearest_school_km", "dist_nearest_hospital_km","dist_nearest_supermarket_km", 
        "dist_nearest_mall_km","dist_nearest_cafe_restaurant_km","school_count_within_3km", "hospital_count_within_3km",
    "supermarket_count_within_3km", "cafe_restaurant_count_within_3km",]

# for accuracy-based quarantine
MAX_VALID_AREA = 1000
MIN_VALID_AREA = 40
MAX_VALID_PRICE=20_000_000
MIN_VALID_LON = 25.0
MAX_VALID_LON=35.0
MIN_VALID_LAT = 22.0
MAX_VALID_LAT=31.0
MIN_VALID_BEDROOMS=1

# for outlier
AREA_LOWER_PERC = 0.01
AREA_UPPER_PERC = 0.95
POI_LOWER_PERC = 0.0
POI_UPPER_PERC = 0.95
#####################       SETUP LOGGING       #####################

os.makedirs(os.path.dirname(CLEANING_LOGGING_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CLEANING_LOG_REPORT_PATH), exist_ok=True)
os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
os.makedirs(os.path.dirname(QUARANTINE_LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(CLEANING_LOGGING_PATH),   # writes to file
        logging.StreamHandler(),               # prints to console
    ],
)

cleaning_log: list[dict] = []


def log_cleaning_action(step: str,rule: str,records_affected: int,action: str,rationale: str) -> None:
    """
    Append one cleaning decision to the in-memory cleaning log.

    Args:
        step:The cleaning dimension
        rule:specific rule applied
        records_affected: Number of rows or values changed.
        action: action that was done.
        rationale: Why this action was chosen.
    """
    cleaning_log.append({
        "step": step,
        "rule": rule,
        "records_affected": records_affected,
        "action": action,
        "rationale": rationale,
    })
    logging.info(f"[LOG] {step} | {rule} | {records_affected} records | {action}")



######################    CLEANING FUNCTIONS    #####################

def remove_irrelevant_columns(df: pd.DataFrame, columns_to_remove: list[str]) -> pd.DataFrame:
    """remove any irrelevant columns that do not contribute to the analysis or modeling."""
    print("Original number of Columns:", df.shape[1])
    df_relevance = df.drop(columns=columns_to_remove, errors="ignore").copy()
    print("Number of Columns after cleaning:", df_relevance.shape[1])

    return df_relevance

def accuracy_rule_based_correction(df:pd.DataFrame) -> pd.DataFrame:
    """Apply accuracy-based corrections to the dataframe."""
    df_cleaned = df.copy()

    incorrect_bedrooms_values_count = df[df["bedrooms"] == "studio"].shape[0]

    df_cleaned["bedrooms"] = df["bedrooms"].replace("studio", 1).astype(int)

    log_cleaning_action(
        step="Accuracy",
        rule="Bedrooms: 'studio' replaced with 1",
        records_affected=incorrect_bedrooms_values_count,
        action="replaced bedrooms with studio value with 1",
        rationale="Studio apartments typically have 1 bedroom equivalent"
    )

    return df_cleaned


def accuracy_qurantine_based_fixing(df, quarantine_log_path=QUARANTINE_LOG_PATH) -> pd.DataFrame:
    area_mask = df["area_value"] > MAX_VALID_AREA
    very_small_area_mask = df["area_value"] < MIN_VALID_AREA
    price_mask = df["price_egp"] > MAX_VALID_PRICE
    lon_mask = (df["lon"] < MIN_VALID_LON) | (df["lon"] > MAX_VALID_LON)
    lat_mask = (df["lat"] < MIN_VALID_LAT) | (df["lat"] > MAX_VALID_LAT)
    bedroom_mask = df["bedrooms"] < MIN_VALID_BEDROOMS

    rejection_mask = (
        area_mask |
        price_mask |
        lon_mask |
        lat_mask |
        very_small_area_mask |
        bedroom_mask
    )

    df_cleaned = df[~rejection_mask].copy()
    df_quarantined = df[rejection_mask].copy()

    if area_mask.sum() > 0:
        log_cleaning_action(
            step="accuracy",
            rule=f"area_value > {MAX_VALID_AREA}",
            records_affected=area_mask.sum(),
            action=f"quarantined non realistic {area_mask.sum()} records",
            rationale="non realistic area values likely indicate data entry errors"
        )

    if price_mask.sum() > 0:
        log_cleaning_action(
            step="accuracy",
            rule=f"price_egp > {MAX_VALID_PRICE}",
            records_affected=price_mask.sum(),
            action=f"quarantined non realistic {price_mask.sum()} records",
            rationale="non realistic price values likely indicate data entry errors"
        )

    if lon_mask.sum() > 0:
        log_cleaning_action(
            step="accuracy",
            rule="lon out of range",
            records_affected=lon_mask.sum(),
            action=f"quarantined invalid {lon_mask.sum()} records",
            rationale="longitude values outside valid range"
        )

    if lat_mask.sum() > 0:
        log_cleaning_action(
            step="accuracy",
            rule="lat out of range",
            records_affected=lat_mask.sum(),
            action=f"quarantined invalid {lat_mask.sum()} records",
            rationale="latitude values outside valid range"
        )

    if very_small_area_mask.sum() > 0:
        log_cleaning_action(
            step="accuracy",
            rule=f"area_value < {MIN_VALID_AREA}",
            records_affected=very_small_area_mask.sum(),
            action=f"quarantined very small area {very_small_area_mask.sum()} records",
            rationale="very small area values likely indicate data entry errors"
        )

    if bedroom_mask.sum() > 0:
        log_cleaning_action(
            step="accuracy",
            rule=f"bedrooms < {MIN_VALID_BEDROOMS}",
            records_affected=bedroom_mask.sum(),
            action=f"quarantined invalid bedrooms {bedroom_mask.sum()} records",
            rationale="invalid bedroom counts for residential properties"
        )

    df_quarantined["rejection_reason"] = (
        df[rejection_mask].apply(
            lambda row: '; '.join([
                f'area_value > {MAX_VALID_AREA}' if row['area_value'] > MAX_VALID_AREA else '',
                f'price_egp > {MAX_VALID_PRICE}' if row['price_egp'] > MAX_VALID_PRICE else '',
                'invalid_lon' if (row['lon'] < MIN_VALID_LON or row['lon'] > MAX_VALID_LON) else '',
                'invalid_lat' if (row['lat'] < MIN_VALID_LAT or row['lat'] > MAX_VALID_LAT) else '',
                f'area_value < {MIN_VALID_AREA}' if row['area_value'] < MIN_VALID_AREA else '',
                'invalid_bedrooms' if row['bedrooms'] < MIN_VALID_BEDROOMS else '',
            ]).strip('; '),
            axis=1
        )
    )

    log_cleaning_action(
        step="accuracy",
        rule="total_quarantined",
        records_affected=rejection_mask.sum(),
        action=f"quarantined {rejection_mask.sum()} records with reasons logged",
        rationale="Allows for later review"
    )


    df_quarantined.to_csv(quarantine_log_path, index=False)
    return df_cleaned


def high_missingness_removal(df):
    cols_to_drop = [
        "is_exclusive",  # about 80% missing and value exists in listing_level
        "amenities"      # about 75% missing and would introduce noise if imputed
    ]

    df_cleaned = df.drop(columns=cols_to_drop).copy()

    log_cleaning_action(
        step="Completeness",
        rule=f"Drop {', '.join(cols_to_drop)} cols high-missingness",
        records_affected=len(df),
        action=f"Dropped columns: {cols_to_drop}",
        rationale="Columns with >50% missing AND not required as model features",
    )
    return df_cleaned






def fix_consistency(df: pd.DataFrame) -> pd.DataFrame:
    df_cleaned = df.copy()
 
    df_cleaned["property_type"] = df_cleaned["property_type"].str.lower().replace({"apartments": "apartment"})
    log_cleaning_action(
        step="consistency", rule="standardize_property_types",
        records_affected=int((df["property_type"].str.lower() == "apartments").sum()),
        action="standardized 'apartments' to 'apartment'",
        rationale="Unifies property type category under the singular form"
    )
    if df_cleaned["property_type"].nunique() == 1:
        df_cleaned.drop(columns=["property_type"], inplace=True)
        log_cleaning_action(
            step="consistency", rule="drop_property_type_if_single_value",
            records_affected=0, action="dropped 'property_type' column",
            rationale="Column is redundant when it contains only a single unique value"
        )
 
    df_cleaned["offering_type"] = df_cleaned["offering_type"].replace({"Residential for Sale": "for-sale"})
    log_cleaning_action(
        step="consistency", rule="standardize_offering_types",
        records_affected=int((df["offering_type"] == "Residential for Sale").sum()),
        action="standardized 'Residential for Sale' to 'for-sale'",
        rationale="Unifies offering type under a concise consistent label"
    )
    if df_cleaned["offering_type"].nunique() == 1:
        df_cleaned.drop(columns=["offering_type"], inplace=True)
        log_cleaning_action(
            step="consistency", rule="drop_offering_type_if_single_value",
            records_affected=0, action="dropped 'offering_type' column",
            rationale="Column is redundant when it contains only a single unique value"
        )
 
    df_cleaned["completion_status"] = df_cleaned["completion_status"].replace({
        "completed_primary": "completed",
        "off_plan_primary": "off_plan"
    })
    log_cleaning_action(
        step="consistency", rule="standardize_completion_status",
        records_affected=int(df["completion_status"].isin(["completed_primary", "off_plan_primary"]).sum()),
        action="standardized 'completed_primary' -> 'completed', 'off_plan_primary' -> 'off_plan'",
        rationale="Unifies completion status values under concise consistent labels"
    )
 
    if "price_period" in df_cleaned.columns and df_cleaned["price_period"].nunique() == 1:
        df_cleaned.drop(columns=["price_period"], inplace=True)
        log_cleaning_action(
            step="consistency", rule="drop_price_period_if_single_value",
            records_affected=0, action="dropped 'price_period' column",
            rationale="Column is redundant when it contains only a single unique value"
        )
 
    if "price_currency" in df_cleaned.columns and df_cleaned["price_currency"].nunique() == 1:
        df_cleaned.drop(columns=["price_currency"], inplace=True)
        log_cleaning_action(
            step="consistency", rule="drop_price_currency_if_single_value",
            records_affected=0, action="dropped 'price_currency' column",
            rationale="Column is redundant when it contains only a single unique value"
        )
 
    df_cleaned["town"] = df_cleaned["town"].str.replace(r'\bCity\b', '', regex=True).str.strip().str.title()
    log_cleaning_action(
        step="consistency", rule="standardize_town_names",
        records_affected=int(df["town"].str.contains(r'\bCity\b', regex=True, na=False).sum()),
        action="removed 'City' suffix and applied title case to town names",
        rationale="Removes redundant 'City' suffix and unifies casing across town names"
    )
 
    df_cleaned["district"] = (
        df_cleaned["district"].astype(str).str.strip().str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\w\s]", "", regex=True)
    )
    log_cleaning_action(
        step="consistency", rule="standardize_district_names",
        records_affected=int(df_cleaned["district"].notna().sum()),
        action="lowercased, stripped, removed extra spaces and punctuation from district names",
        rationale="Unifies district name formatting for consistent grouping"
    )
 
    if "area_unit" in df_cleaned.columns:
        df_cleaned["area_unit"] = df_cleaned["area_unit"].str.lower().str.strip()
        log_cleaning_action(
            step="consistency", rule="standardize_area_unit",
            records_affected=int(df_cleaned["area_unit"].notna().sum()),
            action="lowercased and stripped 'area_unit'",
            rationale="Unifies area unit values for consistency"
        )
        if df_cleaned["area_unit"].nunique() == 1:
            df_cleaned.drop(columns=["area_unit"], inplace=True)
            log_cleaning_action(
                step="consistency", rule="drop_area_unit_if_single_value",
                records_affected=0, action="dropped 'area_unit' column",
                rationale="Column is redundant when it contains only a single unique value"
            )
 
    furnished_mapping = {"NO": "unfurnished", "YES": "furnished", "PARTLY": "partly"}
    log_cleaning_action(
        step="consistency", rule="standardize_furnished",
        records_affected=int(df_cleaned["furnished"].isin(furnished_mapping.keys()).sum()),
        action=f"mapped furnished values: {furnished_mapping}",
        rationale="Unifies furnished status values under consistent lowercase labels"
    )
    df_cleaned["furnished"] = df_cleaned["furnished"].replace(furnished_mapping)
 
    for col in ["is_verified", "is_new_construction", "rera"]:
        if col in df_cleaned.columns and df_cleaned[col].nunique() <= 1:
            df_cleaned.drop(columns=[col], inplace=True)
            log_cleaning_action(
                step="consistency", rule=f"drop_{col}_if_single_value",
                records_affected=0, action=f"dropped '{col}' column",
                rationale="Column is redundant when it contains only a single or no unique value"
            )
 
    return df_cleaned

def fill_district_with_mode(df):
    number_of_missing_districts = int(df["district"].isna().sum())

    if number_of_missing_districts == 0:
        return df,0
        
    df["district"] = df["district"].cat.add_categories(["Unknown"])
    group_mode = df.groupby(['city', 'town'])['district'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    df['district'] = df['district'].fillna(group_mode)

    city_mode = df.groupby('city')['district'].transform(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )
    df['district'] = df['district'].fillna(city_mode)

    df['district'] = df['district'].fillna('Unknown')

    return df,number_of_missing_districts


def fix_missingness(df):
    df_cleaned = df.copy()

    # District Imputation 
    df_cleaned, number_of_missing_districts = fill_district_with_mode(df_cleaned)

    if number_of_missing_districts > 0:
        log_cleaning_action(
            step="completeness",
            rule="fill_district_with_mode",
            action="fill_district_with_mode",
            records_affected=number_of_missing_districts,
            rationale=(
                "Filled missing 'district' values using mode of 'city' and 'town' groups if not empty, "
                "then by 'city' groups, then set remaining to 'Missing' if no mode found"
            ),
        )

    # Furnished Imputation 
    n_furnished_missing = int(df_cleaned["furnished"].isna().sum())

    if n_furnished_missing > 0:
        df_cleaned["furnished"] = df_cleaned["furnished"].cat.add_categories(["Missing"])
        df_cleaned["furnished"] = df_cleaned["furnished"].fillna("Missing")

        log_cleaning_action(
            step="Completeness",
            rule="furnished: fill NaN with 'Missing' category",
            records_affected=n_furnished_missing,
            action="fillna('Missing')",
            rationale=(
                "Missingness is informative — agents who don't disclose furnished status "
                "may represent a distinct listing pattern. 'Missing' lets the model learn from it."
            ),
        )

    # Drop missing completion_status 
    number_missing_completion = int(df_cleaned["completion_status"].isna().sum())
    df_cleaned = df_cleaned.dropna(subset=['completion_status'])

    log_cleaning_action(
        step="Completeness",
        rule="Drop rows with missing completion status",
        records_affected=number_missing_completion,
        action="dropna(subset=['completion_status'])",
        rationale=(
            "Completion status is a critical feature for modeling. "
            "Missingness is relatively low, so dropping is preferable to imputation since very few rows are affected."
        ),
    )

    # Drop missing bathroom 
    number_missing_bathrooms = int(df_cleaned["bathroom"].isna().sum())
    df_cleaned = df_cleaned.dropna(subset=['bathroom'])

    log_cleaning_action(
        step="Completeness",
        rule="Drop rows with missing bathroom count",
        records_affected=number_missing_bathrooms,
        action="dropna(subset=['bathroom'])",
        rationale=(
            "Bathroom count is a critical feature for modeling. "
            "Missingness is relatively low, so dropping is preferable to imputation since very few rows are affected."
        ),
    )

    return df_cleaned



def drop_duplicates(df):
    df_cleaned = df.copy()

    count_of_duplicates = df_cleaned.duplicated().sum()
    df_cleaned.drop_duplicates(inplace=True)

    log_cleaning_action(
        step="Uniqueness",
        rule="drop_exact_duplicates",
        records_affected=count_of_duplicates,
        action="drop_duplicates()",
        rationale=f"Removed {count_of_duplicates} exact duplicate rows to ensure uniqueness of listings."
    )

    logging.info(
        f"Shape after dropping duplicates: {df_cleaned.shape} "
        f"(dropped {count_of_duplicates} rows)"
    )

    return df_cleaned


def apply_clipping(df, column, lower_perc, upper_perc):
    df_clipped = df.copy()
    lower_cap = df_clipped[column].quantile(lower_perc)
    upper_cap = df_clipped[column].quantile(upper_perc)

    number_of_clipped_values = sum((df_clipped[column] < lower_cap) | (df_clipped[column] > upper_cap))

    df_clipped[column] = df_clipped[column].clip(lower=lower_cap, upper=upper_cap)

    return df_clipped, number_of_clipped_values


def handle_outliers_with_clipping(df,clipping_columns=CLIPPING_COLUMNS,log_columns=LOG_COLUMNS):
    df_cleaned = df.copy()

    # ---- Area ----
    df_cleaned, number_of_clipped_values = apply_clipping(df_cleaned, "area_value", AREA_LOWER_PERC, AREA_UPPER_PERC)

    log_cleaning_action(
        step="outlier",
        rule="clip_area_value_outliers",
        records_affected=number_of_clipped_values,
        action=f"Clipped area_value to {AREA_LOWER_PERC*100}th and {AREA_UPPER_PERC*100}th percentiles",
        rationale=(
            "Clipping extreme outliers in 'area_value' reduces their disproportionate influence on the model "
            "while preserving the overall distribution shape."
        )
    )

    # Clipping columns
    for column in clipping_columns:
        if column not in df_cleaned.columns: 
            print(f"Warning: Column '{column}' not found in dataframe. Skipping clipping for this column.")
            continue

        df_cleaned, clipped_numbers = apply_clipping(df_cleaned, column, POI_LOWER_PERC, POI_UPPER_PERC)  # FIXED: df → df_cleaned

        log_cleaning_action(
            step="outlier",
            rule=f"clip_{column}_outliers",
            records_affected=clipped_numbers,
            action=f"Clipped '{column}' to {POI_UPPER_PERC*100}th percentile",
            rationale=(
                f"Clipping extreme outliers in '{column}' reduces their disproportionate"
                "influence on the model while preserving the overall distribution shape. "
            )
        )

    # ---- Count columns ----
    for column in log_columns:  #
        if column not in df_cleaned.columns: 
            print(f"Warning: Column '{column}' not found in dataframe. Skipping log transformation.")
            continue

        df_cleaned[column] = np.log1p(df_cleaned[column])  # FIXED: df → df_cleaned

        log_cleaning_action(
            step="outlier",
            rule=f"log_transform_{column}",
            records_affected=df_cleaned[column].notna().sum(),
            action=f"Applied log1p transformation on '{column}'",
            rationale=(
                f"Log transformation reduces right skewness in '{column}' and compresses extreme values, "
                "making the distribution more stable and less sensitive to outliers while preserving ordering."
            )
        )

    return df_cleaned

#######################    CLEANING PIPELINE       #####################

class DataCleaningPipeline:
    """
    A modular data cleaning pipeline.

    This pipeline applies a series of transformations to clean and
    prepare data for analysis or modeling.
    """
    def __init__(self):
        self.transformations = []

    def add_step(self, func, ** kwargs):
        """Add a transformation step to the pipeline."""
        self.transformations.append((func, kwargs))
        return self

    def fit_transform(self, df):
        """Apply all transformations to the dataframe."""
        result = df.copy()
        for func, kwargs in self.transformations:
            result = func(result, ** kwargs)
        return result




if __name__ == "__main__":
    # Load raw data
    raw_df = pd.read_csv(RAW_DATA_PATH,low_memory=False)
    df=raw_df.copy()
    print("DATA SHAPE")
    print(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]}")
    print()

    # Create and apply cleaning pipeline
    cleaning_pipeline = DataCleaningPipeline()
    cleaning_pipeline.add_step(remove_irrelevant_columns, columns_to_remove=IRRELEVANT_COLUMNS_TO_REMOVE)
    cleaning_pipeline.add_step(accuracy_rule_based_correction)
    cleaning_pipeline.add_step(accuracy_qurantine_based_fixing)
   
    cleaning_pipeline.add_step(fix_consistency)

    cleaning_pipeline.add_step(high_missingness_removal)
    cleaning_pipeline.add_step(fix_missingness)
    cleaning_pipeline.add_step(drop_duplicates)
    cleaning_pipeline.add_step(handle_outliers_with_clipping,\
                                distance_columns=DISTANCE_COLUMNS, count_columns=COUNT_COLUMNS)
    
    clean_df = cleaning_pipeline.fit_transform(df)
    print("FINAL CLEANED DATA SHAPE")
    print(f"Rows: {clean_df.shape[0]:,} | Columns: {clean_df.shape[1]}")

    cleaning_report_df = pd.DataFrame(cleaning_log)
    clean_df.to_csv(CLEANED_DATA_PATH, index=False)
    cleaning_report_df.to_csv(CLEANING_LOG_REPORT_PATH, index=False)



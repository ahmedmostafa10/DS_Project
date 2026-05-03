import os
import logging
import pandas as pd
import numpy as np
from pydantic import BaseModel, ValidationError, Field, field_validator
from typing import Optional


class PropertyRowSchema(BaseModel):
    city: Optional[str] = None
    town: Optional[str] = None
    district: Optional[str] = None
    area_unit: Optional[str] = None
    price_currency: Optional[str] = None
    bedrooms: Optional[float] = None
    bathroom: Optional[float] = None
    area_value: Optional[float] = None

    # OSM Engineered Features
    dist_nearest_school_km: Optional[float] = None
    school_count_within_3km: Optional[float] = None
    dist_nearest_hospital_km: Optional[float] = None
    hospital_count_within_3km: Optional[float] = None
    dist_nearest_supermarket_km: Optional[float] = None
    supermarket_count_within_3km: Optional[float] = None
    dist_nearest_mall_km: Optional[float] = None
    mall_count_within_3km: Optional[float] = None
    dist_nearest_transit_station_km: Optional[float] = None
    transit_station_count_within_3km: Optional[float] = None
    dist_nearest_cafe_restaurant_km: Optional[float] = None
    cafe_restaurant_count_within_3km: Optional[float] = None

    @field_validator("area_value")
    @classmethod
    def check_area_beds_ratio(cls, v, info):
        bedrooms = info.data.get("bedrooms")
        if v is not None and bedrooms is not None:
            if bedrooms > 2 and v < 25:
                raise ValueError("Illogical area vs bedrooms ratio")
        return v

    @field_validator("city", "town", "district", mode="before")
    @classmethod
    def check_capitalization(cls, v):
        if isinstance(v, str) and v != v.title() and v != v.lower() and v != v.upper():
            pass  # Pydantic doesn't trivially group this state across rows but we can do type coercion check
        return v


class DataValidationPipeline:
    """
    Handles data quality validation & profiling.
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.report_summary = {}

    def run_validation(self, input_csv, output_report_csv="data_quality_report.csv"):
        self.logger.info("Starting Data Validation...")
        try:
            df = pd.read_csv(input_csv)
            self.logger.info(f"Data loaded successfully! Shape: {df.shape}")
        except FileNotFoundError:
            self.logger.error(
                f"File {input_csv} not found. Cannot proceed with validation."
            )
            return

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(
            include=["object", "category", "string"]
        ).columns

        self._validate_accuracy(df, numeric_cols)
        self._validate_consistency(df)
        self._validate_completeness(df)
        self._validate_uniqueness(df, categorical_cols)
        self._validate_outliers(df, numeric_cols)
        self._validate_distribution_and_relationships(df, numeric_cols)
        self._generate_charts(df, numeric_cols, categorical_cols, output_report_csv)
        self._validate_context(df)

        self._export_report(output_report_csv)

    def _validate_accuracy(self, df, numeric_cols):
        self.logger.info("--- Accuracy Validation ---")
        accuracy_issues = 0
        acc_details = []
        for col in numeric_cols:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                self.logger.warning(
                    f"Column '{col}' contains {negative_count} negative values."
                )
                accuracy_issues += negative_count
                acc_details.append(f"{col}: {negative_count} negatives")

        if "bedrooms" in df.columns:
            invalid_beds_count = (
                pd.to_numeric(df["bedrooms"], errors="coerce") > 15
            ).sum()
            if invalid_beds_count > 0:
                self.logger.warning(
                    f"Column 'bedrooms' contains {invalid_beds_count} values > 15."
                )
                accuracy_issues += invalid_beds_count
                acc_details.append(f"bedrooms: {invalid_beds_count} values > 15")

        if "bathroom" in df.columns:
            invalid_baths_count = (
                pd.to_numeric(df["bathroom"], errors="coerce") > 15
            ).sum()
            if invalid_baths_count > 0:
                self.logger.warning(
                    f"Column 'bathroom' contains {invalid_baths_count} values > 15."
                )
                accuracy_issues += invalid_baths_count
                acc_details.append(f"bathroom: {invalid_baths_count} values > 15")

        if "area_value" in df.columns:
            invalid_area_count_high = (
                pd.to_numeric(df["area_value"], errors="coerce") > 1000
            ).sum()
            invalid_area_count_low = (
                pd.to_numeric(df["area_value"], errors="coerce") < 10
            ).sum()
            if invalid_area_count_high > 0 or invalid_area_count_low > 0:
                invalid_area_count = invalid_area_count_high + invalid_area_count_low
                self.logger.warning(
                    f"Column 'area_value' contains {invalid_area_count} unrealistic values (> 1000 or < 10 sqm)."
                )
                accuracy_issues += invalid_area_count
                acc_details.append(
                    f"area_value: {invalid_area_count} unrealistic values"
                )

        if "price_egp" in df.columns:
            invalid_price_count = (
                pd.to_numeric(df["price_egp"], errors="coerce") > 500000000
            ).sum()
            if invalid_price_count > 0:
                self.logger.warning(
                    f"Column 'price_egp' contains {invalid_price_count} extremely high values (> 500M EGP)."
                )
                accuracy_issues += invalid_price_count
                acc_details.append(
                    f"price_egp: {invalid_price_count} extreme highs (> 500M EGP)"
                )

        if accuracy_issues == 0:
            self.logger.info("Basic numeric boundaries look accurate.")
            acc_details.append("No boundary violations detected.")

        self.report_summary["Accuracy"] = " | ".join(acc_details)

    def _validate_consistency(self, df):
        self.logger.info("--- Consistency Validation ---")
        consistency_errors = []
        con_details = []
        pydantic_error_count = 0

        for index, row in df.iterrows():
            row_dict = row.replace({np.nan: None}).to_dict()
            try:
                PropertyRowSchema(**row_dict)
            except ValidationError as e:
                pydantic_error_count += 1
                if pydantic_error_count <= 5:
                    self.logger.warning(f"Row {index} failed Pydantic validation: {e}")

        if pydantic_error_count > 0:
            self.logger.warning(
                f"Total rows failing Pydantic consistency validation: {pydantic_error_count}"
            )
            consistency_errors.append("Pydantic Schema/Type/Logic validation failed")
            con_details.append(
                f"{pydantic_error_count} rows failed Pydantic validation"
            )

        cat_cols_to_check = [
            "city",
            "town",
            "district",
            "property_type",
            "listing_type",
        ]
        for col in cat_cols_to_check:
            if col in df.columns:
                lower_vals = df[col].dropna().astype(str).str.lower()
                original_unique = df[col].nunique()
                lower_unique = lower_vals.nunique()
                if original_unique > lower_unique:
                    self.logger.warning(
                        f"[{col}] Found {original_unique - lower_unique} capitalization discrepancies."
                    )
                    consistency_errors.append(f"{col} capitalization mismatch")
                    con_details.append(
                        f"{col}: {original_unique - lower_unique} capitalization discrepancies"
                    )

        if "property_type" in df.columns:
            prop_types = set(
                df["property_type"].dropna().astype(str).str.lower().str.strip()
            )
            pairs_to_check = [
                ("apartment", "apartments"),
                ("villa", "villas"),
                ("townhouse", "townhouses"),
            ]
            mixed_pairs = []
            for sing, plur in pairs_to_check:
                if sing in prop_types and plur in prop_types:
                    mixed_pairs.append(f"{sing}/{plur}")

            if mixed_pairs:
                self.logger.warning(
                    f"[property_type] Found inconsistent singular/plural categories: {', '.join(mixed_pairs)}"
                )
                consistency_errors.append("property_type singular/plural mix")
                con_details.append(f"property_type: mixed {', '.join(mixed_pairs)}")

        if "offering_type" in df.columns:
            off_types = set(
                df["offering_type"].dropna().astype(str).str.lower().str.strip()
            )
            sale_synonyms = {"for-sale", "for sale", "residential for sale", "buy"}
            rent_synonyms = {"for-rent", "for rent", "residential for rent", "rent"}

            found_sale = off_types.intersection(sale_synonyms)
            found_rent = off_types.intersection(rent_synonyms)

            if len(found_sale) > 1:
                self.logger.warning(
                    f"[offering_type] Found inconsistent 'sale' categories: {', '.join(found_sale)}"
                )
                consistency_errors.append("offering_type sale synonyms")
                con_details.append(
                    f"offering_type: mixed sale synonyms ({', '.join(found_sale)})"
                )

            if len(found_rent) > 1:
                self.logger.warning(
                    f"[offering_type] Found inconsistent 'rent' categories: {', '.join(found_rent)}"
                )
                consistency_errors.append("offering_type rent synonyms")
                con_details.append(
                    f"offering_type: mixed rent synonyms ({', '.join(found_rent)})"
                )

        units_cols = ["area_unit", "price_currency"]
        for col in units_cols:
            if col in df.columns:
                unique_units = df[col].dropna().unique()
                if len(unique_units) > 1:
                    self.logger.warning(
                        f"[{col}] Inconsistent units found: {unique_units}"
                    )
                    consistency_errors.append(f"Mixed {col}")
                    con_details.append(f"{col}: Mixed units {list(unique_units)}")

        for col in df.columns:
            col_dropna = df[col].dropna()
            if not col_dropna.empty:
                inferred_type = pd.api.types.infer_dtype(col_dropna)
                if (
                    inferred_type.startswith("mixed")
                    and inferred_type != "mixed-integer-float"
                ):
                    self.logger.warning(
                        f"[{col}] Column contains fundamentally mixed data types (inferred: {inferred_type})."
                    )
                    consistency_errors.append(f"Mixed data types in {col}")
                    con_details.append(f"{col}: mixed-type values ({inferred_type})")

                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    empty_str_count = col_dropna.astype(str).str.strip().eq("").sum()
                    if empty_str_count > 0:
                        self.logger.warning(
                            f"[{col}] Found {empty_str_count} empty string values posing as valid data instead of NaNs."
                        )
                        consistency_errors.append(f"Empty strings in {col}")
                        con_details.append(
                            f"{col}: {empty_str_count} hidden empty strings"
                        )

        if not con_details:
            con_details.append("No consistency issues found")

        self.report_summary["Consistency"] = " | ".join(con_details)

    def _validate_completeness(self, df):
        self.logger.info("--- Completeness Analysis ---")
        comp_details = []
        missing_data = df.isnull().sum()
        completeness_df = pd.DataFrame(
            {
                "Missing Values": missing_data,
                "Percentage (%)": (missing_data / len(df)) * 100,
            }
        )
        completeness_df = completeness_df[
            completeness_df["Missing Values"] > 0
        ].sort_values(by="Percentage (%)", ascending=False)
        self.logger.info(f"Missing values found in {len(completeness_df)} columns.")

        for idx, missing_row in completeness_df.iterrows():
            pct = missing_row["Percentage (%)"]
            row_cnt = missing_row["Missing Values"]
            self.logger.warning(f"Missing Data -> [{idx}]: {row_cnt} rows ({pct:.2f}%)")
            comp_details.append(f"{idx}: {pct:.2f}% missing")

        if not comp_details:
            comp_details.append("No missing data in dataset.")

        self.report_summary["Completeness"] = (
            f"{len(completeness_df)} cols have missing -> All: "
            + " | ".join(comp_details)
        )

    def _validate_uniqueness(self, df, categorical_cols):
        self.logger.info("--- Uniqueness Analysis ---")
        uniq_details = []

        duplicates_count = df.duplicated().sum()
        self.logger.info(f"Exact duplicate rows: {duplicates_count}")
        uniq_details.append(f"Exact duplicates: {duplicates_count}")

        for col in categorical_cols:
            if col in df.columns:
                u_cnt = df[col].nunique()
                self.logger.info(f"Cardinality -> [{col}]: {u_cnt} unique values")
                uniq_details.append(f"{col}: {u_cnt} unique")

        self.report_summary["Uniqueness"] = " | ".join(uniq_details)

    def _validate_outliers(self, df, numeric_cols):
        self.logger.info("--- Outlier Detection: IQR ---")
        iqr_details = []
        for col in numeric_cols:
            if df[col].nunique() > 10 and not col.lower().endswith("id"):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[
                    (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                ]
                if len(outliers) > 0:
                    self.logger.info(
                        f"IQR Outliers -> [{col}]: {len(outliers)} values outside 1.5*IQR boundaries"
                    )
                    iqr_details.append(f"{col}: {len(outliers)} outliers")

        if not iqr_details:
            iqr_details.append("No IQR outliers found.")

        self.report_summary["IQR Outliers"] = " | ".join(iqr_details)

    def _validate_distribution_and_relationships(self, df, numeric_cols):
        self.logger.info("--- Distribution Profiling & Relationships ---")
        dist_details = []
        for col in numeric_cols:
            if df[col].nunique() > 10 and not col.lower().endswith("id"):
                col_data = df[col].dropna()
                if not col_data.empty:
                    skewness = col_data.skew()
                    kurt = col_data.kurtosis()
                    self.logger.info(
                        f"Distribution Profile -> [{col}] Skewness: {skewness:.2f}, Kurtosis: {kurt:.2f}"
                    )
                    dist_details.append(f"{col}: Skew={skewness:.2f}, Kurt={kurt:.2f}")

        if not dist_details:
            dist_details.append("Insufficient numeric data for distribution profile.")

        self.report_summary["Distribution"] = " | ".join(dist_details)

        rel_details = []
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr(method="spearman")
            self.logger.info(
                "Calculated Spearman Correlation Matrix (robust to outliers)."
            )
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.8:
                        self.logger.info(
                            f"High Spearman Correlation: [{corr_matrix.columns[i]}] & [{corr_matrix.columns[j]}] = {val:.2f}"
                        )
                        rel_details.append(
                            f"{corr_matrix.columns[i]}/{corr_matrix.columns[j]}:{val:.2f}"
                        )

        if not rel_details:
            if len(numeric_cols) > 1:
                rel_details.append("No strong (>0.8) Spearman correlations found.")
            else:
                rel_details.append("Insufficient numeric variables for correlation.")

        self.report_summary["Relationships"] = " | ".join(rel_details)

    def _generate_charts(self, df, numeric_cols, categorical_cols, output_report_csv):
        plots_out_dir = os.path.join(os.path.dirname(output_report_csv) or ".", "plots")
        os.makedirs(plots_out_dir, exist_ok=True)

        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 1. Numeric Feature Distribution Profiles (Histograms + KDE)
            cols_to_plot = [
                c
                for c in numeric_cols
                if df[c].nunique() > 10 and not c.lower().endswith("id")
            ]
            if cols_to_plot:
                n_cols = len(cols_to_plot)
                grid_cols = min(3, n_cols)
                grid_rows = (n_cols + grid_cols - 1) // grid_cols
                fig, axes = plt.subplots(
                    grid_rows, grid_cols, figsize=(6 * grid_cols, 5 * grid_rows)
                )

                if n_cols == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()

                for i, col in enumerate(cols_to_plot):
                    data_to_plot = df[col].dropna()

                    # Address extreme scaling issues for price and area by dropping top 2% extreme outliers for the plot
                    if col in ["price_egp", "area_value"]:
                        q_high = data_to_plot.quantile(0.98)
                        data_to_plot = data_to_plot[data_to_plot <= q_high]
                        axes[i].set_title(f"Distribution Profile: {col} (≤ P98)")
                    else:
                        axes[i].set_title(f"Distribution Profile: {col}")

                    sns.histplot(
                        data_to_plot, kde=True, bins=30, ax=axes[i], color="skyblue"
                    )

                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])

                plt.tight_layout()
                dist_path = os.path.join(
                    plots_out_dir, "numeric_distribution_profiles.png"
                )
                plt.savefig(dist_path)
                plt.close(fig)
                self.logger.info(
                    f"📊 Numeric distribution profiles saved to {dist_path}"
                )

            # 2. Categorical Bar Chart (Class Distribution)
            if len(categorical_cols) > 0:
                target_cat = (
                    "property_type"
                    if "property_type" in df.columns
                    else categorical_cols[0]
                )
                plt.figure(figsize=(10, 6))
                df[target_cat].value_counts().head(10).plot(
                    kind="bar", color="coral", edgecolor="black"
                )
                plt.title(f"Class Distribution: {target_cat}")
                plt.ylabel("Frequency")
                plt.xticks(rotation=45)
                plt.tight_layout()
                bar_path = os.path.join(
                    plots_out_dir, f"class_distribution_{target_cat}.png"
                )
                plt.savefig(bar_path)
                plt.close()
                self.logger.info(f"📊 Class distribution bar chart saved to {bar_path}")

            # 3. Spearman Correlation Heatmap
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                corr_matrix = df[numeric_cols].corr(method="spearman")
                sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5)
                plt.title("Spearman Correlation Matrix")
                plt.tight_layout()
                heatmap_path = os.path.join(
                    plots_out_dir, "correlation_heatmap_spearman.png"
                )
                plt.savefig(heatmap_path)
                plt.close()
                self.logger.info(f"📊 Correlation heatmap saved to {heatmap_path}")

        except ImportError:
            self.logger.warning(
                "matplotlib or seaborn not installed. Skipping chart generation."
            )

    def _validate_context(self, df):
        self.logger.info("--- Description & Context ---")
        target_col = "price"
        if target_col in df.columns:
            self.logger.info(f"Target Definition: Target defined as '{target_col}'")
            self.report_summary["Target Definition"] = (
                f"Target defined as '{target_col}'."
            )
        else:
            self.logger.info("Target Definition: Target variable undefined.")
            self.report_summary["Target Definition"] = "Target variable undefined."

        engineered_cols = [
            col
            for col in df.columns
            if any(kw in col.lower() for kw in ["osm", "dist", "nearest", "count"])
        ]
        self.logger.info(
            f"Engineered Features: {len(engineered_cols)} features found {engineered_cols}"
        )
        self.report_summary["Engineered Features"] = (
            f"{len(engineered_cols)} engineered features identified."
        )

    def _export_report(self, output_report_csv):
        report_df = pd.DataFrame(
            list(self.report_summary.items()),
            columns=["Dimension", "Findings / Summary"],
        )
        report_df.to_csv(output_report_csv, index=False)
        self.logger.info(f"✅ Data Quality Report exported to {output_report_csv}")

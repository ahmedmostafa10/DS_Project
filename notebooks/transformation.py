import logging
import os
import ast
import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from scripts.paths import DATA_DIR, REPORTS_DIR


class DataTransformationPipeline:
    """
    Applies feature transformation and engineering to model-ready data.
    Principles covered:
    - Feature selection
    - Binning/discretization
    - Feature interactions
    - Feature encoding (one-hot, label, frequency, binary, rare, optional target)
    - Feature scaling (standard, min-max, robust)
    - Documentation via report export
    """

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.report_summary = {}

    def run_transformation(
        self,
        input_csv,
        output_csv,
        output_report_csv=str(REPORTS_DIR / "transformation_report.csv"),
        target_col: Optional[str] = None,
    ):
        self.logger.info("Starting Data Transformation...")
        self.logger.info(f"Input file: {input_csv}")
        self.logger.info(f"Output file: {output_csv}")
        self.logger.info(f"Report file: {output_report_csv}")

        try:
            df = pd.read_csv(input_csv, low_memory=False)
        except FileNotFoundError:
            self.logger.error(f"Input file not found: {input_csv}")
            return

        # Prevent mixed-type surprises in known messy columns.
        if "listing_id" in df.columns:
            df["listing_id"] = df["listing_id"].astype("string")
        if "amenities" in df.columns:
            df["amenities"] = df["amenities"].astype("string")

        self.logger.info(f"Transformation input shape: {df.shape}")

        if target_col is None:
            inferred_target = self._infer_target_column(df)
            if inferred_target is not None:
                target_col = inferred_target
                self.logger.info(f"Inferred target column for target encoding: {target_col}")

        self.logger.info("Applying amenities feature extraction...")
        df = self._apply_amenities_features(df)
        self.logger.info("Applying feature scaling...")
        df = self._apply_binning(df)
        self.logger.info("Applying feature interactions...")
        df = self._apply_scaling(df)
        self.logger.info("Applying feature encoding...")
        df = self._apply_encoding(df, target_col=target_col)
        self.logger.info("Applying binning/discretization...")
        df = self._apply_feature_interactions(df)
        self.logger.info("Applying feature selection...")
        df = self._feature_selection(df)

        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(output_csv, index=False, encoding="utf-8")
        self.logger.info(f"Saved transformed dataset with shape: {df.shape}")

        self._export_report(output_report_csv)
        self.logger.info(f"Transformation complete. Output saved to: {output_csv}")

    def _apply_amenities_features(self, df):
        self.logger.info("--- Amenities Feature Extraction ---")
        details = []

        if "amenities" not in df.columns:
            self.logger.info("Skipping amenities extraction: 'amenities' column not found")
            details.append("Skipped: amenities column not found")
            self.report_summary["Amenities Features"] = " | ".join(details)
            return df

        raw_series = df["amenities"]

        parsed_amenities = []
        amenity_vocab = set()
        missing_flags = []

        for value in raw_series:
            items, is_missing = self._parse_amenities_value(value)
            parsed_amenities.append(items)
            missing_flags.append(1 if is_missing else 0)
            amenity_vocab.update(items)

        sorted_amenities = sorted(amenity_vocab)
        self.logger.info(f"Unique amenities extracted: {len(sorted_amenities)}")
        if sorted_amenities:
            self.logger.info(
                "Sample extracted amenities: "
                + ", ".join(sorted_amenities[:25])
                + (" ..." if len(sorted_amenities) > 25 else "")
            )

        # Binary amenity features
        amenity_feature_names = []
        for amenity in sorted_amenities:
            safe_name = re.sub(r"[^a-z0-9]+", "_", amenity.lower()).strip("_")
            if not safe_name:
                continue
            col_name = f"amenity_{safe_name}"
            df[col_name] = [1 if amenity in row_items else 0 for row_items in parsed_amenities]
            amenity_feature_names.append(col_name)

        # Count feature + missing flag
        df["amenities_count"] = [len(row_items) for row_items in parsed_amenities]
        df["missing_amenities"] = missing_flags

        missing_count = int(sum(missing_flags))
        self.logger.info(f"Added binary amenity columns: {len(amenity_feature_names)}")
        self.logger.info(f"Added 'amenities_count' and 'missing_amenities' columns")
        self.logger.info(f"Rows with missing amenities: {missing_count}")

        if amenity_feature_names:
            self.logger.info(
                "Amenity feature columns sample: "
                + ", ".join(amenity_feature_names[:25])
                + (" ..." if len(amenity_feature_names) > 25 else "")
            )

        details.append(f"Amenity binary columns: {len(amenity_feature_names)}")
        details.append(f"Rows missing amenities: {missing_count}")
        details.append("Added: amenities_count, missing_amenities")
        self.report_summary["Amenities Features"] = " | ".join(details)
        return df

    def _parse_amenities_value(self, value):
        if pd.isna(value):
            return [], True

        if isinstance(value, list):
            cleaned = [str(x).strip() for x in value if str(x).strip()]
            return sorted(set(cleaned)), len(cleaned) == 0

        text = str(value).strip()
        if not text:
            return [], True

        lowered = text.lower()
        if lowered in {"nan", "none", "null", "[]", "{}"}:
            return [], True

        parsed_items = []

        # Try Python/JSON-like list literal first.
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                parsed_items = [str(x).strip() for x in parsed if str(x).strip()]
            elif isinstance(parsed, str) and parsed.strip():
                parsed_items = [parsed.strip()]
        except Exception:
            parsed_items = []

        # Fallback to delimiter split.
        if not parsed_items:
            chunks = re.split(r"[|,;]", text)
            parsed_items = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]

        if not parsed_items:
            return [], True

        return sorted(set(parsed_items)), False


    def _apply_scaling(self, df):
        self.logger.info("--- Feature Scaling ---")
        details = []

        profile_source = df
        profile_path = DATA_DIR / "cleaned_data.csv"
        if profile_path.exists():
            try:
                profile_source = pd.read_csv(profile_path, low_memory=False)
                self.logger.info(f"Profiling scaling from {profile_path}")
            except Exception as exc:
                self.logger.warning(f"Could not read {profile_path}. Falling back to current dataframe. Error: {exc}")

        candidate_cols = profile_source.select_dtypes(include=[np.number]).columns.tolist()
        candidate_cols = [c for c in candidate_cols if c in df.columns]
        self.logger.info(
            "Scaling candidate columns: "
            + (", ".join(candidate_cols[:25]) + (" ..." if len(candidate_cols) > 25 else ""))
        )

        excluded_cols = []
        scaled_cols = []

        for col in candidate_cols:
            lower = col.lower()
            if "id" in lower or lower.endswith("_id"):
                excluded_cols.append(col)
                continue

            series = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            non_null = series.dropna()
            if non_null.empty:
                excluded_cols.append(col)
                continue

            unique_vals = set(non_null.unique())
            if len(unique_vals) <= 2 and unique_vals.issubset({0, 1}):
                excluded_cols.append(col)
                continue

            if non_null.nunique() <= 1:
                excluded_cols.append(col)
                continue

            q1 = non_null.quantile(0.25)
            q3 = non_null.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_ratio = ((non_null < lower_bound) | (non_null > upper_bound)).mean()
            else:
                outlier_ratio = 0.0

            skewness = abs(non_null.skew()) if non_null.shape[0] > 2 else 0.0

            if outlier_ratio > 0.05 or skewness > 1.5:
                scaler = RobustScaler()
                scaler_name = "RobustScaler"
            elif non_null.min() >= 0 and (
                non_null.max() > 1
                or "count" in lower
                or "dist" in lower
                or "price" in lower
                or "area" in lower
            ):
                scaler = MinMaxScaler()
                scaler_name = "MinMaxScaler"
            else:
                scaler = StandardScaler()
                scaler_name = "StandardScaler"

            filled = series.fillna(non_null.median())
            transformed = scaler.fit_transform(filled.values.reshape(-1, 1)).flatten()
            transformed_series = pd.Series(transformed, index=df.index)
            transformed_series[series.isna()] = np.nan
            df[col] = transformed_series

            scaled_cols.append((col, scaler_name))
            self.logger.info(
                f"Scaled column '{col}' using {scaler_name} (skew={skewness:.2f}, outliers={outlier_ratio:.2%})"
            )

        self.logger.info(f"Columns scaled: {len(scaled_cols)}")
        self.logger.info(f"Columns skipped from scaling: {len(excluded_cols)}")
        if scaled_cols:
            scaled_names = [name for name, _ in scaled_cols]
            self.logger.info(
                "Scaled columns list: "
                + (", ".join(scaled_names[:25]) + (" ..." if len(scaled_names) > 25 else ""))
            )
        if excluded_cols:
            self.logger.info(
                "Skipped columns list: "
                + (", ".join(excluded_cols[:25]) + (" ..." if len(excluded_cols) > 25 else ""))
            )

        if scaled_cols:
            details.append(f"Scaled columns: {len(scaled_cols)}")
            details.append(
                "Scaler choices: "
                + ", ".join(f"{name}->{scaler}" for name, scaler in scaled_cols[:15])
                + (" ..." if len(scaled_cols) > 15 else "")
            )
        else:
            details.append("No columns required scaling")

        details.append(f"Skipped columns: {len(excluded_cols)}")
        self.report_summary["Scaling"] = " | ".join(details)
        return df


    def _apply_binning(self, df):
        self.logger.info("--- Binning / Discretization ---")
        details = []

        bin_targets = {
            "price_egp": "price_bin",
            "area_value": "area_bin",
            "bedrooms": "bedrooms_bin",
        }

        for source_col, bin_col in bin_targets.items():
            self.logger.info(f"Evaluating binning source column: {source_col}")
            if source_col not in df.columns:
                self.logger.info(f"Skipping binning for missing column: {source_col}")
                continue

            source_data = pd.to_numeric(df[source_col], errors="coerce")
            if source_data.nunique(dropna=True) < 4:
                self.logger.info(f"Skipping binning for low-cardinality column: {source_col}")
                continue

            try:
                df[bin_col] = pd.qcut(source_data, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
                details.append(f"{source_col} -> {bin_col} (quartiles)")
                self.logger.info(f"Created binned column: {bin_col}")
            except ValueError:
                self.logger.info(f"Could not bin '{source_col}' because of insufficient distribution.")

        if not details:
            details.append("No binned features created")

        self.report_summary["Binning"] = " | ".join(details)
        return df

    def _apply_feature_interactions(self, df):
        self.logger.info("--- Feature Interactions ---")
        details = []

        if {"price_egp", "area_value"}.issubset(df.columns):
            price = pd.to_numeric(df["price_egp"], errors="coerce")
            area = pd.to_numeric(df["area_value"], errors="coerce")
            safe_area = area.replace(0, np.nan)
            df["price_per_sqm"] = price / safe_area
            details.append("Created price_per_sqm")
            self.logger.info("Created interaction feature: price_per_sqm")
        else:
            self.logger.info("Skipping interaction price_per_sqm: requires columns price_egp and area_value")

        if {"bedrooms", "bathroom"}.issubset(df.columns):
            beds = pd.to_numeric(df["bedrooms"], errors="coerce")
            baths = pd.to_numeric(df["bathroom"], errors="coerce")
            df["total_rooms"] = beds.fillna(0) + baths.fillna(0)
            df["bed_bath_ratio"] = beds / baths.replace(0, np.nan)
            details.append("Created total_rooms")
            details.append("Created bed_bath_ratio")
            self.logger.info("Created interaction features: total_rooms, bed_bath_ratio")
        else:
            self.logger.info("Skipping interactions total_rooms/bed_bath_ratio: requires bedrooms and bathroom")

        if not details:
            details.append("No interaction features created")

        self.report_summary["Feature Interactions"] = " | ".join(details)
        return df

    def _apply_encoding(self, df, target_col: Optional[str] = None):
        self.logger.info("--- Feature Encoding ---")
        details = []

        target_encoding_cols = [c for c in ["town", "district"] if c in df.columns]
        if target_col and target_col in df.columns:
            encoded_target_cols = []
            for col in target_encoding_cols:
                encoded = self._target_encode_column(df, col, target_col)
                if encoded is not None:
                    df[f"{col}_te"] = encoded
                    df.drop(columns=[col], inplace=True)
                    encoded_target_cols.append(col)
                    self.logger.info(f"Target encoded column '{col}' into '{col}_te' using cross-validation")
            if encoded_target_cols:
                details.append(f"Target encoded columns: {len(encoded_target_cols)}")
        elif target_encoding_cols:
            self.logger.info(
                "Skipping target encoding for town/district because no target column was provided or found"
            )

        # Binary encoding for boolean and bool-like columns.
        bool_cols = [c for c in df.columns if df[c].dtype == bool]
        bool_like_cols = []
        for col in df.columns:
            if col in bool_cols:
                continue
            values = set(df[col].dropna().astype(str).str.lower().unique())
            if values and values.issubset({"true", "false", "0", "1", "yes", "no"}):
                bool_like_cols.append(col)

        for col in bool_cols:
            df[col] = df[col].astype(int)
        for col in bool_like_cols:
            mapped = (
                df[col]
                .astype(str)
                .str.lower()
                .map({"true": 1, "yes": 1, "1": 1, "false": 0, "no": 0, "0": 0})
            )
            df[col] = mapped

        if bool_cols or bool_like_cols:
            details.append(f"Binary encoded columns: {len(bool_cols) + len(bool_like_cols)}")
        self.logger.info(
            f"Binary encoding candidates -> bool: {len(bool_cols)}, bool-like: {len(bool_like_cols)}"
        )
        if bool_cols:
            self.logger.info("Bool columns encoded: " + ", ".join(bool_cols))
        if bool_like_cols:
            self.logger.info("Bool-like columns encoded: " + ", ".join(bool_like_cols))

        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        self.logger.info(
            "Categorical columns considered: "
            + (", ".join(cat_cols[:25]) + (" ..." if len(cat_cols) > 25 else ""))
        )

        cat_cols = [c for c in cat_cols if c not in target_encoding_cols]

        # Rare encoding for categorical levels below 1% frequency.
        rare_encoded_cols = []
        for col in cat_cols:
            freq = df[col].value_counts(normalize=True, dropna=True)
            rare_levels = freq[freq < 0.01].index
            if len(rare_levels) > 0:
                df[col] = df[col].where(~df[col].isin(rare_levels), other="Rare")
                rare_encoded_cols.append(col)

        if rare_encoded_cols:
            details.append(f"Rare encoded columns (<1%): {len(rare_encoded_cols)}")
        self.logger.info(f"Rare-encoded categorical columns: {len(rare_encoded_cols)}")
        if rare_encoded_cols:
            self.logger.info("Rare-encoded columns list: " + ", ".join(rare_encoded_cols))

        # Frequency encoding for high-cardinality categories.
        freq_encoded_cols = []
        for col in cat_cols:
            if col not in df.columns:
                continue
            nunique = df[col].nunique(dropna=True)
            if nunique > 15:
                freq_map = df[col].value_counts(normalize=True)
                df[f"{col}_freq"] = df[col].map(freq_map)
                freq_encoded_cols.append(col)

        if freq_encoded_cols:
            details.append(f"Frequency encoded columns: {len(freq_encoded_cols)}")
        self.logger.info(f"Frequency-encoded categorical columns: {len(freq_encoded_cols)}")
        if freq_encoded_cols:
            self.logger.info("Frequency-encoded columns list: " + ", ".join(freq_encoded_cols))

        # Optional target encoding for supervised settings.
        if target_col and target_col in df.columns:
            target_numeric = pd.to_numeric(df[target_col], errors="coerce")
            if target_numeric.notna().any():
                te_cols = []
                for col in cat_cols:
                    if col == target_col:
                        continue
                    if col in df.columns and 2 <= df[col].nunique(dropna=True) <= 50:
                        means = df.groupby(col)[target_numeric.name if hasattr(target_numeric, "name") else target_col].mean()
                        df[f"{col}_target_mean"] = df[col].map(means)
                        te_cols.append(col)
                if te_cols:
                    details.append(f"Target encoded columns: {len(te_cols)}")
                    self.logger.info("Target-encoded columns list: " + ", ".join(te_cols))
            else:
                details.append("Target encoding skipped (target not numeric)")
        else:
            details.append("Target encoding skipped (no target column provided)")

        # One-hot encode low-cardinality categories.
        updated_cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        one_hot_cols = [c for c in updated_cat_cols if df[c].nunique(dropna=True) <= 8]
        if one_hot_cols:
            df = pd.get_dummies(df, columns=one_hot_cols, drop_first=False)
            details.append(f"One-hot encoded columns: {len(one_hot_cols)}")
        self.logger.info(f"One-hot encoded columns count: {len(one_hot_cols)}")
        if one_hot_cols:
            self.logger.info("One-hot encoded columns list: " + ", ".join(one_hot_cols))

        # Label encode ordinal bins where one-hot is not ideal.
        label_cols = [c for c in df.columns if c.endswith("_bin") and (df[c].dtype == "object" or str(df[c].dtype).startswith("category"))]
        for col in label_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        if label_cols:
            details.append(f"Label encoded bin columns: {len(label_cols)}")
        self.logger.info(f"Label-encoded bin columns count: {len(label_cols)}")
        if label_cols:
            self.logger.info("Label-encoded bin columns list: " + ", ".join(label_cols))

        if not details:
            details.append("No encoding operations performed")

        self.report_summary["Encoding"] = " | ".join(details)
        return df

    def _infer_target_column(self, df):
        candidate_columns = ["completion_status", "is_verified", "is_premium"]
        for column in candidate_columns:
            if column in df.columns:
                return column
        return None

    def _target_encode_column(self, df, column, target_col, n_splits=5, smoothing=10.0):
        if column not in df.columns or target_col not in df.columns:
            return None

        data = df[[column, target_col]].copy()
        target_series = data[target_col]

        if pd.api.types.is_numeric_dtype(target_series):
            target_values = pd.to_numeric(target_series, errors="coerce")
        else:
            target_values = pd.Series(pd.factorize(target_series.astype("string"))[0], index=data.index)

        if target_values.isna().all():
            return None

        non_null = target_values.dropna()
        if non_null.nunique() <= 1:
            self.logger.info(f"Skipping target encoding for '{column}' because target has no variance")
            return None

        feature = data[column].astype("string").fillna("__missing__")
        global_mean = float(non_null.mean())
        encoded = pd.Series(index=df.index, dtype=float)

        class_counts = non_null.nunique()
        use_stratified = class_counts <= 10 and set(non_null.unique()).issubset({0, 1})
        splitter = (
            StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            if use_stratified
            else KFold(n_splits=n_splits, shuffle=True, random_state=42)
        )

        split_input = non_null if not use_stratified else non_null.astype(int)

        for train_idx, val_idx in splitter.split(feature.loc[non_null.index], split_input):
            train_index = feature.loc[non_null.index].iloc[train_idx].index
            val_index = feature.loc[non_null.index].iloc[val_idx].index

            train_feature = feature.loc[train_index]
            train_target = target_values.loc[train_index]
            mapping = train_target.groupby(train_feature).agg(["mean", "count"])
            mapping["encoded"] = (
                mapping["mean"] * mapping["count"] + global_mean * smoothing
            ) / (mapping["count"] + smoothing)
            fold_encoded = feature.loc[val_index].map(mapping["encoded"]).fillna(global_mean)
            encoded.loc[val_index] = fold_encoded

        full_mapping = target_values.groupby(feature).agg(["mean", "count"])
        full_mapping["encoded"] = (
            full_mapping["mean"] * full_mapping["count"] + global_mean * smoothing
        ) / (full_mapping["count"] + smoothing)
        encoded = encoded.fillna(feature.map(full_mapping["encoded"])).fillna(global_mean)

        return encoded

    

    def _feature_selection(self, df):
        self.logger.info("--- Feature Selection ---")
        details = []

        before_count = len(df.columns)

        missing_ratio = df.isna().mean()
        high_missing_cols = missing_ratio[missing_ratio > 0.95].index.tolist()
        if high_missing_cols:
            df = df.drop(columns=high_missing_cols)
            details.append(f"Dropped high-missing columns (>95%): {len(high_missing_cols)}")
            self.logger.info("High-missing columns dropped: " + ", ".join(high_missing_cols))

        constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if constant_cols:
            df = df.drop(columns=constant_cols)
            details.append(f"Dropped constant columns: {len(constant_cols)}")
            self.logger.info("Constant columns dropped: " + ", ".join(constant_cols))

        after_count = len(df.columns)
        removed_count = before_count - after_count

        details.append(f"Columns before selection: {before_count}")
        details.append(f"Columns after selection: {after_count}")
        details.append(f"Total removed columns: {removed_count}")

        if removed_count == 0:
            self.logger.info("Feature selection did not remove any columns")
        else:
            self.logger.info(f"Feature selection removed {removed_count} columns")

        self.report_summary["Feature Selection"] = " | ".join(details)
        return df
    def _export_report(self, output_report_csv):
        self.logger.info("Exporting transformation report...")
        report_df = pd.DataFrame(
            list(self.report_summary.items()),
            columns=["Dimension", "Findings / Summary"],
        )

        report_dir = os.path.dirname(output_report_csv)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)

        report_df.to_csv(output_report_csv, index=False)
        self.logger.info(f"Transformation report exported to {output_report_csv}")


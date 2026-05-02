import os

import tomllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV

with open("./configs/config.toml", "rb") as f:
    config = tomllib.load(f)

CLEANED_DATA_PATH = config["paths"]["cleaned_data_path"]
TRANSFORMATION_LOG_REPORT_PATH = config["paths"]["transformation_log_report_path"]
TRANSFORMATION_LOGGING_PATH = config["paths"]["transformation_logging_path"]
PROCESSED_DIR = config["paths"]["processed_data_dir"]
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(TRANSFORMATION_LOG_REPORT_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler(TRANSFORMATION_LOGGING_PATH),
        logging.StreamHandler(),
    ],
)


class FeatureTransformation:

    def __init__(self):
        self.transformation_log_report: list[dict] = []
        self.district_stats = None
        self.town_stats = None
        self.global_stats = None
        self.area_median = None
        self.rfecv = None
        self.selected_rfecv = None

    def log_transformation_action(self, stage: str, column: str, action: str, reason: str) -> None:
        self.transformation_log_report.append({
            "stage": stage,
            "column": column,
            "action": action,
            "reason": reason,
        })
        logging.info(f"[LOG] {stage} | {column} | {action} | {reason}")

    def load_data(self):
        df = pd.read_csv(CLEANED_DATA_PATH)
        return df

    def create_target(self, df):
      
        df['price_egp_bin'] = pd.qcut(df['price_egp'], q=3, labels=[0, 1, 2])
        return df

    def split_data(self, df):
        X = df.drop(['price_egp', 'price_egp_bin'], axis=1)
        y = df['price_egp_bin']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["split"]["test_size"],
            stratify=y,
            random_state=config["split"]["random_state"]
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=config["split"]["val_size"],
            stratify=y_train,
            random_state=config["split"]["random_state"]
        )

        return X_train.copy(), X_val.copy(), X_test.copy(), y_train, y_val, y_test
        

    def scale_features(self, X_train, X_val, X_test):


        for col in ['lat', 'lon']:
            scaler = RobustScaler()
            X_train[col] = scaler.fit_transform(X_train[[col]])
            X_val[col] = scaler.transform(X_val[[col]])
            X_test[col] = scaler.transform(X_test[[col]])
           
            self.log_transformation_action(
                stage="Feature Scaling",
                column=col,
                action="RobustScaler",
                reason="To handle outliers and scale the feature to a similar range as other features."
            )

        
        scaler = StandardScaler()
        X_train['area_value'] = scaler.fit_transform(X_train[['area_value']])
        X_val['area_value'] = scaler.transform(X_val[['area_value']])
        X_test['area_value'] = scaler.transform(X_test[['area_value']])
      
        self.log_transformation_action(
            stage="Feature Scaling",
            column="area_value",
            action="StandardScaler",
            reason="To scale the feature to have mean 0 and variance 1, which can help some models perform better."
        )

        standard_dist_cols = ['dist_nearest_mall_km', 'dist_nearest_transit_station_km']
        for c in standard_dist_cols:
            scaler = StandardScaler()
            X_train[c] = scaler.fit_transform(X_train[[c]])
            X_val[c] = scaler.transform(X_val[[c]])
            X_test[c] = scaler.transform(X_test[[c]])
   
            self.log_transformation_action(
                stage="Feature Scaling",
                column=c,
                action="StandardScaler",
                reason="To scale the feature to have mean 0 and variance 1, which can help some models perform better."
            )

        robust_dist_cols = ['dist_nearest_school_km', 'dist_nearest_hospital_km',
                            'dist_nearest_supermarket_km', 'dist_nearest_cafe_restaurant_km']
        for c in robust_dist_cols:
            if c in X_train.columns:
                scaler = RobustScaler()
                X_train[c] = scaler.fit_transform(X_train[[c]])
                X_val[c] = scaler.transform(X_val[[c]])
                X_test[c] = scaler.transform(X_test[[c]])
              
                self.log_transformation_action(
                    stage="Feature Scaling",
                    column=c,
                    action="RobustScaler",
                    reason="To handle outliers and scale the feature to a similar range as other features."
                )


        for col in X_train.columns:
            if col.endswith('count_within_3km'):
                scaler = RobustScaler()
                X_train[col] = scaler.fit_transform(X_train[[col]])
                X_val[col] = scaler.transform(X_val[[col]])
                X_test[col] = scaler.transform(X_test[[col]])
               
                self.log_transformation_action(
                    stage="Feature Scaling",
                    column=col,
                    action="RobustScaler",
                    reason="To handle outliers and scale the feature to a similar range as other features."
                )

        return X_train, X_val, X_test

    def encode_features(self, X_train, X_val, X_test):
        listing_map = config["encoding"]["listing_level"]


        for df_ in [X_train, X_val, X_test]:
            df_["listing_level"] = df_["listing_level"].map(listing_map)
        self.log_transformation_action(
            stage="Feature Encoding",
            column="listing_level",
            action="Ordinal Encoding",
            reason="To convert the categorical feature into a numerical format, while preserving the ordinal relationship between the categories."
        )

        cat_cols = config["encoding"]["cat_col"]
        encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)
        encoded_train = encoder.fit_transform(X_train[cat_cols])
        encoded_val = encoder.transform(X_val[cat_cols])
        encoded_test = encoder.transform(X_test[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        encoded_train = pd.DataFrame(encoded_train, columns=encoded_cols, index=X_train.index)
        encoded_val = pd.DataFrame(encoded_val, columns=encoded_cols, index=X_val.index)
        encoded_test = pd.DataFrame(encoded_test, columns=encoded_cols, index=X_test.index)
        X_train = pd.concat([X_train.drop(columns=cat_cols), encoded_train], axis=1)
        X_val = pd.concat([X_val.drop(columns=cat_cols), encoded_val], axis=1)
        X_test = pd.concat([X_test.drop(columns=cat_cols), encoded_test], axis=1)
        self.log_transformation_action(
            stage="Feature Encoding",
            column=", ".join(cat_cols),
            action="One-Hot Encoding",
            reason="To convert the categorical features into a numerical format, without assuming any ordinal relationship between the categories."
        )

        freq_cols = config["encoding"]["freq_col"]
        for col in freq_cols:
            freq_map = X_train[col].value_counts(normalize=True)
            for df in [X_train, X_val, X_test]:
                df[col] = df[col].map(freq_map).fillna(0)
        self.log_transformation_action(
            stage="Feature Encoding",
            column="city, town, district",
            action="Frequency Encoding",
            reason="To convert the categorical features into a numerical format, while capturing the frequency of each category which may be related to the target variable."
        )

        for df in [X_train, X_val, X_test]:
            bool_cols = df.select_dtypes(bool).columns
            df[bool_cols] = df[bool_cols].astype(int)
        self.log_transformation_action(
            stage="Feature Encoding",
            column=bool_cols.tolist(),
            action="Boolean to Integer Encoding",
            reason="To convert boolean features into a numerical format"
        )

        return X_train, X_val, X_test

    def add_arithmetic_features(self, df):
        df = df.copy()
        df["area_per_bedroom"] = df["area_value"] / (df["bedrooms"])
        df["area_per_bathroom"] = df["area_value"] / (df["bathroom"])
        df["bathroom_per_bedroom"] = df["bathroom"] / (df["bedrooms"])
        df["total_rooms"] = df["bathroom"] + df["bedrooms"]

        count_cols =config["data"]["count_columns"]
        df["total_services_count_3km"] = df[count_cols].sum(axis=1)

        dist_cols = config["data"]["distance_columns"]
        df["avg_distance_services_3km"] = df[dist_cols].mean(axis=1)
        df["min_distance_services_3km"] = df[dist_cols].min(axis=1)
        df["accessibility_score"] = sum(1 / (df[col] + 1) for col in dist_cols)

        return df

    def fit_location_stats(self, df):
        df = df.copy()
        district_stats = df.groupby("district").agg({
            "area_value": "mean",
            "bedrooms": "mean",
            "bathroom": "mean",
            "total_services_count_3km": "mean"
        }).rename(columns={
            "area_value": "district_avg_area",
            "bedrooms": "district_avg_bedrooms",
            "bathroom": "district_avg_bathroom",
            "total_services_count_3km": "district_avg_services"
        })

        town_stats = df.groupby("town").agg({
            "area_value": "mean",
            "bedrooms": "mean",
            "bathroom": "mean"
        }).rename(columns={
            "area_value": "town_avg_area",
            "bedrooms": "town_avg_bedrooms",
            "bathroom": "town_avg_bathroom"
        })

        global_stats = {
            "area_value": df["area_value"].mean(),
            "bedrooms": df["bedrooms"].mean(),
            "bathroom": df["bathroom"].mean(),
            "total_services_count_3km": df["total_services_count_3km"].mean()
        }

        return district_stats, town_stats, global_stats

    def apply_location_stats(self, df, district_stats, town_stats, global_stats):
        df = df.copy()
        original_index = df.index
        df = df.merge(district_stats, on="district", how="left")
        df = df.merge(town_stats, on="town", how="left")
        df.index = original_index

        df["district_avg_area"] = (
            df["district_avg_area"]
            .fillna(df["town_avg_area"])
            .fillna(global_stats["area_value"])
        )
        df["district_avg_bedrooms"] = (
            df["district_avg_bedrooms"]
            .fillna(df["town_avg_bedrooms"])
            .fillna(global_stats["bedrooms"])
        )
        df["district_avg_bathroom"] = (
            df["district_avg_bathroom"]
            .fillna(df["town_avg_bathroom"])
            .fillna(global_stats["bathroom"])
        )
        df["district_avg_services"] = df["district_avg_services"].fillna(global_stats["total_services_count_3km"])
        df["town_avg_area"] = df["town_avg_area"].fillna(global_stats["area_value"])
        df["town_avg_bedrooms"] = df["town_avg_bedrooms"].fillna(global_stats["bedrooms"])
        df["town_avg_bathroom"] = df["town_avg_bathroom"].fillna(global_stats["bathroom"])

        df["area_vs_district_avg"] = (df["area_value"] - df["district_avg_area"]) / df["district_avg_area"]
        df["bedrooms_vs_district_avg"] = (df["bedrooms"] - df["district_avg_bedrooms"]) / df["district_avg_bedrooms"]
        df["bathroom_vs_district_avg"] = (df["bathroom"] - df["district_avg_bathroom"]) / df["district_avg_bathroom"]

        return df

    def apply_binary_features(self, df, area_median):
        df = df.copy()
        df["is_large_house"] = df["area_value"] > area_median
        df["is_small_house"] = df["area_value"] < area_median
        df["near_school"] = df["dist_nearest_school_km"] < 1
        df["near_mall"] = df["dist_nearest_mall_km"] < 2
        df["high_quality_listing"] = (df["is_premium"] == 1) | (df["is_featured"] == 1)
        return df

    def add_feature_interactions(self, X_train, X_val, X_test):
        X_train = self.add_arithmetic_features(X_train)
        X_val = self.add_arithmetic_features(X_val)
        X_test = self.add_arithmetic_features(X_test)

        self.district_stats, self.town_stats, self.global_stats = self.fit_location_stats(X_train)
        X_train = self.apply_location_stats(X_train, self.district_stats, self.town_stats, self.global_stats)
        X_val = self.apply_location_stats(X_val, self.district_stats, self.town_stats, self.global_stats)
        X_test = self.apply_location_stats(X_test, self.district_stats, self.town_stats, self.global_stats)

        self.area_median = X_train["area_value"].median()
        X_train = self.apply_binary_features(X_train, self.area_median)
        X_val = self.apply_binary_features(X_val, self.area_median)
        X_test = self.apply_binary_features(X_test, self.area_median)

        return X_train, X_val, X_test

    def find_correlated_features(self, corr_matrix, threshold=0.9):
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        correlated_pairs = []
        for column in upper.columns:
            if column == 'target':
                continue
            corr_feats = upper.index[abs(upper[column]) > threshold].tolist()
            for cf in corr_feats:
                if cf != 'target':
                    correlated_pairs.append({
                        'feature_A': column,
                        'feature_B': cf,
                        'correlation': round(corr_matrix.loc[column, cf], 3),
                    })
        return pd.DataFrame(correlated_pairs)

    def select_features(self, X_train, X_val, X_test, y_train):
        selector = VarianceThreshold(
        threshold=config["feature_selection"]["variance_threshold"]
    )
        X_train_var = selector.fit_transform(X_train)
        selected_features = X_train.columns[selector.get_support()]
        X_train = pd.DataFrame(X_train_var, columns=selected_features, index=X_train.index)
        X_val = X_val[selected_features]
        X_test = X_test[selected_features]
       
        self.log_transformation_action(
            stage="Feature Selection",
            column=f"Number of selected features: {len(selected_features)}",
            action="Variance Thresholding",
            reason="To remove features that have very low variance, which are unlikely to contribute to model performance."
        )

        corr = X_train.corrwith(y_train).abs().sort_values(ascending=False)
       

        selected_corr = corr[corr > config["feature_selection"]["correlation_threshold"]].index.tolist()
       
        X_train = X_train[selected_corr]
        X_val = X_val[selected_corr]
        X_test = X_test[selected_corr]
        self.log_transformation_action(
            stage="Feature Selection",
            column=f"Number of selected features: {len(selected_corr)}",
            action="Correlation Thresholding",
            reason="To remove features that have a very low correlation with the target variable, which are unlikely to contribute to model performance."
        )

        df_corr = X_train.copy()
        df_corr["target"] = y_train
        corr_matrix = df_corr.corr()
       
        correlated_features = self.find_correlated_features(corr_matrix, threshold=config["feature_selection"]["multicollinearity_threshold"])
      

        features_to_remove = set()
        for _, row in correlated_features.iterrows():
            feat1, feat2 = row['feature_A'], row['feature_B']
            corr_1_t = abs(corr_matrix.loc[feat1, 'target'])
            corr_2_t = abs(corr_matrix.loc[feat2, 'target'])
            features_to_remove.add(feat2 if corr_1_t > corr_2_t else feat1)

        features_to_keep = [c for c in df_corr.columns if c not in features_to_remove and c != 'target']
    

        X_train = X_train[features_to_keep]
        X_val = X_val[features_to_keep]
        X_test = X_test[features_to_keep]
        self.log_transformation_action(
            stage="Feature Selection",
            column=f"Number of selected features: {len(features_to_keep)}",
            action="Correlation Analysis",
            reason="To remove highly correlated features while retaining the one with the strongest correlation to the target variable, thus reducing multicollinearity and improving model performance."
        )

    

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.rfecv = RFECV(
        estimator=model,
        step=config["feature_selection"]["rfecv_step"],
        cv=config["feature_selection"]["rfecv_cv"],
        scoring=config["feature_selection"]["rfecv_scoring"],
        n_jobs=-1
    )
        self.rfecv.fit(X_train, y_train)

        self.selected_rfecv = X_train.columns[self.rfecv.support_]
       

        X_train = X_train[self.selected_rfecv]
        X_val = X_val[self.selected_rfecv]
        X_test = X_test[self.selected_rfecv]
        self.log_transformation_action(
            stage="Feature Selection",
            column=f"Number of selected features: {len(self.selected_rfecv)}",
            action="RFECV",
            reason="To automatically select the optimal number of features based on cross-validated model performance, while eliminating less important features."
        )

        return X_train, X_val, X_test

    def save_outputs(self, X_train, X_val, X_test, y_train, y_val, y_test):
       
        # print the statistics like col rows num and 
        print (f"Final feature set: {X_train.columns.tolist()}")
        print (f"Number of features selected: {X_train.shape[1]}")
        print (f"Number of training samples: {X_train.shape[0]}")

        #also for test and val
        print (f"Number of validation samples: {X_val.shape[0]}")
        print (f"Number of test samples: {X_test.shape[0]}")

        

        transformation_log_df = pd.DataFrame(self.transformation_log_report)
        transformation_log_df.to_csv(config["paths"]["transformation_log_report_path"], index=False)

        train_df = X_train.copy()
        train_df["target"] = y_train
        train_df.to_csv(config["paths"]["train_path"], index=False)

        val_df = X_val.copy()
        val_df["target"] = y_val
        val_df.to_csv(config["paths"]["validation_path"], index=False)

        test_df = X_test.copy()
        test_df["target"] = y_test
        test_df.to_csv(config["paths"]["test_path"], index=False)


def pipeline():
    ft = FeatureTransformation()

    df = ft.load_data()
    df = ft.create_target(df)
    X_train, X_val, X_test, y_train, y_val, y_test = ft.split_data(df)

    X_train, X_val, X_test = ft.scale_features(X_train, X_val, X_test)
    X_train, X_val, X_test = ft.encode_features(X_train, X_val, X_test)
    X_train, X_val, X_test = ft.add_feature_interactions(X_train, X_val, X_test)
    X_train, X_val, X_test = ft.select_features(X_train, X_val, X_test, y_train)

    ft.save_outputs(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    pipeline()
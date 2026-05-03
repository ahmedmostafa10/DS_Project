import logging
import os

import pandas as pd

from .feature_transformation import FeatureTransformation


class DataTransformationPipeline:
    """
    Handles data transformation pipeline orchestration.
    Wraps FeatureTransformation to provide pipeline-level functionality.
    """

    def __init__(self):
        """Initialize the transformation pipeline."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.report_summary = {}
        self.feature_transformer = FeatureTransformation()

    def run_transformation(self, input_csv, output_csv, output_report_csv=None, target_col=None):
        """
        Run the complete transformation pipeline.

        Args:
            input_csv (str): Path to input CSV file
            output_csv (str): Path to output CSV file
            output_report_csv (str, optional): Path to transformation report CSV
            target_col (str, optional): Name of the target column (price, price_egp, etc.)
        """
        self.logger.info("Starting Data Transformation Pipeline...")

        try:
            # Load input data
            df = pd.read_csv(input_csv)
            self.logger.info(f"Data loaded successfully! Shape: {df.shape}")
        except FileNotFoundError:
            self.logger.error(f"File {input_csv} not found. Cannot proceed with transformation.")
            return
        except Exception as e:
            self.logger.error(f"Error loading file {input_csv}: {str(e)}")
            return

        try:
            # Create output directory if needed
            if output_csv:
                os.makedirs(
                    os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".",
                    exist_ok=True,
                )

                # Create target variable if needed
                # Try to detect price column
                price_col = target_col or "price_egp"
                if price_col not in df.columns:
                    # Try alternative column names
                    for col in ["price_egp", "price"]:
                        if col in df.columns:
                            price_col = col
                            break

                # Only create target if it doesn't already exist and we found a price column
                if "price_egp_bin" not in df.columns and price_col in df.columns:
                    self.logger.info(f"Creating target variable from {price_col} column...")
                    try:
                        # Temporarily rename column if needed for create_target to work
                        if price_col != "price_egp":
                            df["price_egp"] = df[price_col]
                        df = self.feature_transformer.create_target(df)
                        self.logger.info("Target variable created successfully")
                    except Exception as e:
                        self.logger.warning(f"Could not create target variable: {str(e)}")

                # Save the transformed data
                df.to_csv(output_csv, index=False)
                self.logger.info(f"Transformed data saved to {output_csv}")

            # Save transformation report if requested
            if output_report_csv:
                os.makedirs(
                    (
                        os.path.dirname(output_report_csv)
                        if os.path.dirname(output_report_csv)
                        else "."
                    ),
                    exist_ok=True,
                )

                # Save transformation log report
                if self.feature_transformer.transformation_log_report:
                    log_df = pd.DataFrame(self.feature_transformer.transformation_log_report)
                    log_df.to_csv(output_report_csv, index=False)
                    self.logger.info(f"Transformation report saved to {output_report_csv}")
                else:
                    # Create empty report file
                    pd.DataFrame(columns=["stage", "column", "action", "reason"]).to_csv(
                        output_report_csv, index=False
                    )

            self.logger.info("Data Transformation Pipeline completed successfully!")

        except Exception as e:
            self.logger.error(f"Error during transformation: {str(e)}")
            raise

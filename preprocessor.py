import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, List, Union, Any
import logging
import gc
from tqdm import tqdm

# Assuming ProgressLogger is defined elsewhere or will be provided.
# If not, you might want to mock it or replace its calls with print statements.
class ProgressLogger:
    def __init__(self, total_steps: int):
        self.total_steps = total_steps
        self.current_step = 0
        logging.info(f"ProgressLogger initialized with {total_steps} steps.")

    def update(self, message: str):
        self.current_step += 1
        logging.info(f"Step {self.current_step}/{self.total_steps}: {message}")
        print(f"Progress: {self.current_step}/{self.total_steps} - {message}") # For console output


class DataPreprocessor:
    def __init__(self, batch_size: int = 10000):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.batch_size = batch_size
        self.progress = None
        self.feature_groups = {
            'packet_features': {
                'required': {'Total Fwd Packets', 'Total Backward Packets'},
                'derived': ['packet_ratio'] # Removed packet_rate as it needs Flow Duration
            },
            'byte_features': {
                'required': {'Total Length of Fwd Packets', 'Total Length of Bwd Packets'},
                'derived': ['byte_ratio'] # Removed byte_rate as it needs Flow Duration
            },
            'flow_features': {
                'required': {'Flow Duration'},
                'derived': ['packet_rate', 'byte_rate'] # These are derived in enhance_features based on Flow Duration
            }
        }
        self.absolutely_required = {'Label'} # This seems to be for fitting, not transform
        self.feature_names = None
        self.cat_columns = None

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms new incoming data using the fitted preprocessor.
        This method assumes `preprocess_data` (or fit on training data) has been called previously
        to set `self.feature_names` and `self.cat_columns` and fit the scaler.
        """
        try:
            if self.feature_names is None:
                raise ValueError("Preprocessor has not been fitted. Call `preprocess_data` on training data first.")
            if self.cat_columns is None: # Ensure cat_columns is set
                logging.warning("Categorical columns were not identified during fit. Proceeding without one-hot encoding.")
                # If cat_columns is not set, we assume no categorical columns for transform
                # In a robust system, you might want to infer them or expect them to be passed.

            # Make a copy to avoid modifying the original DataFrame passed in
            df = df.copy()

            # Process numeric and handle missing values for all columns
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = self.process_numeric_column(df[col])
                else: # Assume non-numeric are categorical, fill with 'NA' if None
                    df[col] = df[col].fillna('NA').astype(str) # Ensure it's string type for get_dummies

            # Drop known irrelevant columns if they exist in the input
            cols_to_drop_inference = ["Flow Bytes/s", "Flow Packets/s", "Flow ID",
                                      "Src IP", "Dst IP", "Timestamp"]
            existing_cols_to_drop = [col for col in cols_to_drop_inference if col in df.columns]
            if existing_cols_to_drop:
                df = df.drop(columns=existing_cols_to_drop)

            # Enhance features (e.g., calculate ratios and rates)
            df = self.enhance_features(df)

            # Process categorical variables using the identified categorical columns from fit
            if self.cat_columns:
                # Align categorical columns: get_dummies might create new columns
                # or miss columns if the input data doesn't have all categories seen during fit.
                # We need to ensure that the dummy columns are consistent with self.feature_names.
                df_dummies = pd.get_dummies(df, columns=self.cat_columns, dtype=np.int8)
                
                # Drop original categorical columns if they are not part of self.feature_names (already dummified)
                numeric_cols_in_df = [col for col in df_dummies.columns if pd.api.types.is_numeric_dtype(df_dummies[col])]
                df = df_dummies[numeric_cols_in_df] # Keep only numeric and new dummy cols

            # Reindex to ensure all features present during fitting are present, fill missing with 0
            # This is crucial for consistent input to the model.
            df = df.reindex(columns=[col for col in self.feature_names if col in df.columns], fill_value=0)
            
            # Add back any columns from self.feature_names that are not in df after reindex (e.g., missing dummy vars)
            missing_features = set(self.feature_names) - set(df.columns)
            for feature in missing_features:
                df[feature] = 0
            
            # Ensure the order of columns matches the training order
            df = df[self.feature_names]

            scaled_data = []
            # Process in batches for large dataframes
            for i in range(0, len(df), self.batch_size):
                batch = df.iloc[i:i + self.batch_size]
                # We use transform, not fit_transform, as scaler is already fitted
                scaled_batch = self.scaler.transform(batch)
                scaled_data.append(scaled_batch)
                del batch
                gc.collect() # Explicitly call garbage collector

            return pd.DataFrame(np.vstack(scaled_data), columns=self.feature_names)

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise

    def get_feature_names_out(self) -> List[str]:
        if self.feature_names is None:
            raise ValueError("Feature names are not available. Call `preprocess_data` first.")
        return list(self.feature_names)

    def get_column_dtypes(self, df: pd.DataFrame) -> Dict[str, str]:
        """Helper to get string representation of dtypes."""
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def process_numeric_column(self, series: pd.Series) -> pd.Series:
        """
        Handles missing values in a numeric series by imputing with chunk-wise mean.
        """
        if series.isnull().any():
            # Use series.mean() for the entire series to avoid potential issues with empty chunks
            # or simply rely on SimpleImputer if a more robust solution is needed.
            # For this context, we'll keep the chunk-wise logic but simplify.
            
            # Calculate overall mean for imputation
            overall_mean = series.mean()
            series = series.fillna(overall_mean)
        return series # No need for transpose on a Series

    def preprocess_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fits the preprocessor and transforms the training data.
        This method identifies categorical columns, fits the scaler and label encoder.
        """
        try:
            self.progress = ProgressLogger(total_steps=5) # Initialize ProgressLogger
            df = df.copy() # Work on a copy of the dataframe

            self.progress.update("Validating input data")
            if 'Label' not in df.columns:
                raise ValueError("Label column missing from dataset")
            
            # Ensure 'Label' is handled as a string for LabelEncoder
            df['Label'] = df['Label'].astype(str)

            self.progress.update("Processing columns (filling NaNs)")
            # First pass to fill NaNs for numeric and categorical columns before feature engineering
            for col in tqdm(df.columns, desc="Processing columns"):
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = self.process_numeric_column(df[col])
                else:
                    df[col] = df[col].fillna('NA').astype(str) # Fill non-numeric with 'NA' and ensure string type

            self.progress.update("Feature enhancement")
            df = self.enhance_features(df) # Apply feature engineering

            self.progress.update("Dropping irrelevant columns and separating features/labels")
            cols_to_drop_fit = ["Flow Bytes/s", "Flow Packets/s", "Flow ID",
                                "Src IP", "Dst IP", "Timestamp"]
            existing_cols_to_drop = [col for col in cols_to_drop_fit if col in df.columns]
            if existing_cols_to_drop:
                df = df.drop(columns=existing_cols_to_drop)
            
            X = df.drop('Label', axis=1)
            y = df['Label'].copy()

            self.progress.update("Processing categorical variables (One-Hot Encoding)")
            # Identify categorical columns that are not numeric and are in X
            self.cat_columns = [col for col in X.columns if not pd.api.types.is_numeric_dtype(X[col])]
            
            if self.cat_columns:
                X = pd.get_dummies(X, columns=self.cat_columns, dtype=np.int8)
                # After get_dummies, the original cat_columns are gone, replaced by new dummy columns.
                # We need to update self.cat_columns if we want to refer to the *new* categorical columns,
                # but typically self.cat_columns should store the names of columns *before* encoding.

            self.progress.update("Scaling features (fitting scaler)")
            scaled_data = []
            # Fit the scaler using batches for potentially very large datasets
            for i in tqdm(range(0, len(X), self.batch_size), desc="Fitting & Scaling batches"):
                batch = X.iloc[i:i + self.batch_size]
                if i == 0: # Fit and transform the first batch
                    scaled_batch = self.scaler.fit_transform(batch)
                else: # Only transform subsequent batches
                    scaled_batch = self.scaler.transform(batch)
                scaled_data.append(scaled_batch)
                del batch
                gc.collect()

            X_scaled_df = pd.DataFrame(np.vstack(scaled_data), columns=X.columns)
            
            # Store the feature names after all transformations (including one-hot encoding)
            self.feature_names = X_scaled_df.columns.tolist()

            # Fit and transform the labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            return X_scaled_df, y_encoded

        except Exception as e:
            logging.error(f"Error in data preprocessing: {str(e)}")
            raise

    def enhance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds derived features like packet ratios and byte ratios/rates.
        Handles division by zero by replacing 0 with 1 for denominators and setting result to 0.
        """
        enhanced_dfs = []
        for start_idx in range(0, len(df), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(df))
            batch = df.iloc[start_idx:end_idx].copy()

            # Ensure required columns exist, if not, create them with 0s
            for group, features in self.feature_groups.items():
                for col in features['required']:
                    if col not in batch.columns:
                        batch[col] = 0.0 # Add missing required columns as floats

            # Calculate derived features
            # Packet Ratio
            if 'Total Fwd Packets' in batch.columns and 'Total Backward Packets' in batch.columns:
                batch['packet_ratio'] = np.divide(
                    batch['Total Fwd Packets'],
                    batch['Total Backward Packets'].replace(0, 1), # Replace 0 with 1 to avoid division by zero
                    out=np.zeros(len(batch), dtype=float),
                    where=batch['Total Backward Packets'] != 0 # Only calculate where denominator is not 0
                )
            else:
                batch['packet_ratio'] = 0.0 # Default if required cols are missing

            # Byte Ratio
            if 'Total Length of Fwd Packets' in batch.columns and 'Total Length of Bwd Packets' in batch.columns:
                batch['byte_ratio'] = np.divide(
                    batch['Total Length of Fwd Packets'],
                    batch['Total Length of Bwd Packets'].replace(0, 1), # Replace 0 with 1 to avoid division by zero
                    out=np.zeros(len(batch), dtype=float),
                    where=batch['Total Length of Bwd Packets'] != 0
                )
            else:
                batch['byte_ratio'] = 0.0 # Default if required cols are missing

            # Rates (depend on Flow Duration)
            # Handle Flow Duration potentially being zero to avoid division by zero
            flow_duration_safe = batch['Flow Duration'].replace(0, 1e-6) # Replace 0 with a small number to avoid div by zero

            if 'Total Fwd Packets' in batch.columns and 'Total Backward Packets' in batch.columns and 'Flow Duration' in batch.columns:
                batch['packet_rate'] = (batch['Total Fwd Packets'] + batch['Total Backward Packets']) / flow_duration_safe
            else:
                batch['packet_rate'] = 0.0

            if 'Total Length of Fwd Packets' in batch.columns and 'Total Length of Bwd Packets' in batch.columns and 'Flow Duration' in batch.columns:
                batch['byte_rate'] = (batch['Total Length of Fwd Packets'] + batch['Total Length of Bwd Packets']) / flow_duration_safe
            else:
                batch['byte_rate'] = 0.0
            
            # Ensure all derived columns are present, even if not calculated due to missing inputs
            for group, features in self.feature_groups.items():
                for derived_col in features['derived']:
                    if derived_col not in batch.columns:
                        batch[derived_col] = 0.0 # Initialize if not created

            enhanced_dfs.append(batch)

            del batch
            gc.collect()

        return pd.concat(enhanced_dfs, axis=0, ignore_index=True)

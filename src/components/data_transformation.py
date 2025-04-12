import sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # Initialize transformers
            numeric_transformer = StandardScaler()
            logging.info("Transformers Initialized: StandardScaler-MinMaxScaler")

            # Load schema configurations
            num_features = self._schema_config['num_features']            
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ("StandardScaler", numeric_transformer, num_features),
                    # ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e
    
    def get_data_transformer_object_target(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered get_data_transformer_object_target method of DataTransformation class")

        try:
            # Initialize transformers
            min_max_scaler = MinMaxScaler()
            logging.info("Transformers Initialized: MinMaxScaler")

            # Load schema configurations
            mm_columns = self._schema_config['mm_columns']       
            logging.info("Cols loaded from schema.")

            # Creating preprocessor pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    # ("StandardScaler", numeric_transformer, num_features),
                    ("MinMaxScaler", min_max_scaler, mm_columns)
                ],
                remainder='passthrough'  # Leaves other columns as they are
            )

            # Wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Pipeline Ready!!")
            logging.info("Exited get_data_transformer_object_target method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise MyException(e, sys) from e

    def _map_datetime_column(self, df):
        """Map datetime column."""
        logging.info("Mapping 'Gender' column to binary values")
        df['Datetime']=pd.to_datetime(df.Datetime)

        df.sort_values(by='Datetime', ascending=True, inplace=True)

        chronological_order = df['Datetime'].is_monotonic_increasing

        time_diffs = df['Datetime'].diff()
        equidistant_timestamps = time_diffs.nunique() == 1
        return df

    def _create_features(self, df):
        """
        Create time series features based on time series index.
        """

        df = df.copy()
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['year'] = df.index.year
        df['season'] = df['month'] % 12 // 3 + 1
        df['dayofyear'] = df.index.dayofyear
        df['dayofmonth'] = df.index.day
        df['weekofyear'] = df.index.isocalendar().week
        
        # Additional features
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = (df['dayofmonth'] == 1).astype(int)
        df['is_month_end'] = (df['dayofmonth'] == df.index.days_in_month).astype(int)
        df['is_quarter_start'] = (df['dayofmonth'] == 1) & (df['month'] % 3 == 1).astype(int)
        df['is_quarter_end'] = (df['dayofmonth'] == df.groupby(['year', 'quarter'])['dayofmonth'].transform('max'))
        
        # Additional features
        df['is_working_day'] = df['dayofweek'].isin([0, 1, 2, 3, 4]).astype(int)
        df['is_business_hours'] = df['hour'].between(9, 17).astype(int)
        df['is_peak_hour'] = df['hour'].isin([8, 12, 18]).astype(int)
        
        # Minute-level features
        df['minute_of_day'] = df['hour'] * 60 + df['minute']
        df['minute_of_week'] = (df['dayofweek'] * 24 * 60) + df['minute_of_day']
        
        return df.astype(float)

    def _drop_id_column(self, df):
        """Drop the 'id' column if it exists."""
        logging.info("Dropping 'id' column")
        drop_col = self._schema_config['drop_columns']
        if drop_col in df.columns:
            df = df.drop(drop_col, axis=1)
        return df

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !!!")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # Load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=TARGET_COLUMN, axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=TARGET_COLUMN, axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # Apply custom transformations in specified sequence
            input_feature_train_df = self._drop_id_column(input_feature_train_df)
            input_feature_train_df = self._map_datetime_column(input_feature_train_df)
            input_feature_train_df = input_feature_train_df.set_index('Datetime')
            input_feature_train_df = self._create_features(input_feature_train_df)

            input_feature_test_df = self._drop_id_column(input_feature_test_df)
            input_feature_test_df = self._map_datetime_column(input_feature_test_df)
            input_feature_test_df = input_feature_test_df.set_index('Datetime')
            input_feature_test_df = self._create_features(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")
            preprocessor_target = self.get_data_transformer_object_target()
            logging.info("Got the preprocessor object for target")

            print("Columns in the DataFrame before transformation:")
            print(input_feature_train_df.columns.tolist())
            print(target_feature_train_df.columns.tolist())
            # print("Columns expected by preprocessor:")
            # print("num_features:", num_features)
            # print("mm_columns:", mm_columns)

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            target_feature_train_arr = preprocessor_target.fit_transform(target_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            target_feature_test_arr = preprocessor_target.fit_transform(target_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            # logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            # smt = SMOTEENN(sampling_strategy="minority")
            # input_feature_train_final, target_feature_train_final = smt.fit_resample(
            #     input_feature_train_arr, target_feature_train_df
            # )
            # input_feature_test_final, target_feature_test_final = smt.fit_resample(
            #     input_feature_test_arr, target_feature_test_df
            # )
            # logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

        except Exception as e:
            raise MyException(e, sys) from e
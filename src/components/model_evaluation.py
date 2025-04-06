from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.exception import MyException
from src.constants import TARGET_COLUMN
from src.logger import logging
from src.utils.main_utils import load_object
import sys
import pandas as pd
from typing import Optional
from src.entity.s3_estimator import Proj1Estimator
from dataclasses import dataclass

@dataclass
class EvaluateModelResponse:
    trained_model_mae_score: float
    best_model_mae_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:

    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise MyException(e, sys) from e

    def get_best_model(self) -> Optional[Proj1Estimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path=self.model_eval_config.s3_model_key_path
            proj1_estimator = Proj1Estimator(bucket_name=bucket_name,
                                               model_path=model_path)

            if proj1_estimator.is_model_present(model_path=model_path):
                return proj1_estimator
            return None
        except Exception as e:
            raise  MyException(e,sys)
        
    def _map_datetime_column(self, df):
        """Map datetime column."""
        logging.info("Mapping 'datetime' column")
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
        if "_id" in df.columns:
            df = df.drop("_id", axis=1)
        return df

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]

            logging.info("Test data loaded and now transforming it for prediction...")

            x = self._drop_id_column(x)
            x = self._map_datetime_column(x)
            x = x.set_index('Datetime')
            x = self._create_features(x)

            trained_model = load_object(file_path=self.model_trainer_artifact.trained_model_file_path)
            logging.info("Trained model loaded/exists.")
            trained_model_mae_score = self.model_trainer_artifact.metric_artifact.MAE
            trained_model_mse_score = self.model_trainer_artifact.metric_artifact.MSE
            logging.info(f"MAE for this model: {trained_model_mae_score}")
            logging.info(f"MAE for this model: {trained_model_mse_score}")

            best_model_mae_score=None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                y_hat_best_model = best_model.predict(x)
                best_model_mae_score = mean_absolute_error(y, y_hat_best_model)
                logging.info(f"MAE-Production Model: {best_model_mae_score}, MAE-New Trained Model: {trained_model_mae_score}")
            
            tmp_best_model_score = 0 if best_model_mae_score is None else best_model_mae_score
            result = EvaluateModelResponse(trained_model_mae_score=trained_model_mae_score,
                                           best_model_mae_score=best_model_mae_score,
                                           is_model_accepted=trained_model_mae_score > tmp_best_model_score,
                                           difference=trained_model_mae_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            print("------------------------------------------------------------------------------------------------")
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise MyException(e, sys) from e
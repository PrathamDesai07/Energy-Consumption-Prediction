import sys
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import xgboost as xgb
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, load_object, save_object
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.entity.estimator import MyModel

class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_model_object_and_report(self, train: np.array, test: np.array) -> Tuple[object, object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a RandomForestClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training XGboost with specified parameters")

            # Splitting the train and test data into features and target variables
            x_train, y_train, x_test, y_test = train[:, :-3], train[:, -3:], test[:, :-3], test[:, -3:]
            logging.info("train-test split done.")

            # Initialize RandomForestClassifier with specified parameters
            param_grid = {
            'objective': ['reg:squarederror'],
            'max_depth': [2],
            'learning_rate': [0.1],
            'subsample': [0.5],
            'n_estimators': [1000],
            'min_child_weight': [2],
            'booster': ['gbtree']
            }
            # param_grid = {
            # 'objective': ['reg:squarederror'],
            # 'max_depth': [2, 5, 7],
            # 'learning_rate': [0.1],
            # 'subsample': [0.5, 0.7],
            # 'n_estimators': [1000, 1500],
            # 'min_child_weight': [1, 2],
            # 'booster': ['gbtree']
            # }

            # Create the XGBoost model object
            xgb_model = xgb.XGBRegressor()


            # Create the GridSearchCV 
            grid_search = GridSearchCV(xgb_model, param_grid, cv=7, scoring='neg_mean_squared_error', verbose=3)
            
            logging.info("Model training going on...")
            # Fit the GridSearchCV object to the training data
            grid_search.fit(x_train, y_train)

            # Print the best set of hyperparameters and the corresponding score
            print("Best set of hyperparameters: ", grid_search.best_params_)
            print("Best score: ", grid_search.best_score_)
            logging.info("Model training done.")

            # Predictions and evaluation metrics
            y_pred = grid_search.best_estimator_.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            logging.info("Model Scoring done.")
            # Creating metric artifact
            metric_artifact = ClassificationMetricArtifact(MSE=mse, MAE=mae)
            model =  grid_search.best_estimator_
            return model, metric_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load transformed train and test data
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, metric_artifact = self.get_model_object_and_report(train=train_arr, test=test_arr)
            logging.info("Model object and artifact loaded.")
            
            # Load preprocessing object
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            logging.info(self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing obj loaded.")

            # Check if the model's accuracy meets the expected threshold
            # if accuracy_score(train_arr[:, -3:], trained_model.predict(train_arr[:, :-3])) < self.model_trainer_config.expected_accuracy:
            #     logging.info("No model found with score above the base score")
            #     raise Exception("No model found with score above the base score")

            # Save the final model object that includes both preprocessing and the trained model
            logging.info("Saving new model as performace is better than previous one.")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(self.model_trainer_config.trained_model_file_path, my_model)
            logging.info("Saved final model object that includes both preprocessing and the trained model")

            # Create and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e, sys) from e
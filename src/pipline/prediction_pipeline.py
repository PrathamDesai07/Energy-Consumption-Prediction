import sys
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class EnergyData:
    def __init__(self,
                Datetime,
                Temperature,
                Humidity,
                WindSpeed,
                GeneralDiffuseFlows,
                DiffuseFlows,
                ):
        """
        Datetime,Temperature,Humidity,WindSpeed,GeneralDiffuseFlows,DiffuseFlows
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.Datetime = Datetime
            self.Temperature = Temperature
            self.Humidity = Humidity
            self.WindSpeed = WindSpeed
            self.GeneralDiffuseFlows = GeneralDiffuseFlows
            self.DiffuseFlows = DiffuseFlows


        except Exception as e:
            raise MyException(e, sys) from e

    def get_energy_input_data_frame(self)-> DataFrame:
        """
        This function returns a DataFrame from USvisaData class input
        """
        try:
            
            vehicle_input_dict = self.get_energy_data_as_dict()
            return DataFrame(vehicle_input_dict)
        
        except Exception as e:
            raise MyException(e, sys) from e
            
    def _create_features(self, df: DataFrame):
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

    def get_energy_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        Datetime,Temperature,Humidity,WindSpeed,GeneralDiffuseFlows,DiffuseFlows
        """
        logging.info("Entered get_energy_data_as_dict method as EnergyData class")

        try:
            input_data = {
                "Datetime": [self.Datetime],
                "Temperature": [self.Temperature],
                "Humidity": [self.Humidity],
                "WindSpeed": [self.WindSpeed],
                "GeneralDiffuseFlows": [self.GeneralDiffuseFlows],
                "DiffuseFlows": [self.DiffuseFlows],
            }

            logging.info("Created vehicle data dict")
            logging.info("Exited get_vehicle_data_as_dict method as EnergyData class")
            return input_data

        except Exception as e:
            raise MyException(e, sys) from e

class EnergyDataClassifier:
    def __init__(self,prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of EnergyDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of EnergyDataClassifier class")
            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result =  model.predict(dataframe)
            
            return result
        
        except Exception as e:
            raise MyException(e, sys)
import os, sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor


from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Split into train and test input data.')
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:,-1],  #last data as y_train value
                test_array[:,:-1],
                test_array[:,-1], #y_test
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbor Regressor": KNeighborsRegressor(),
                "XGBRegressor": XGBRFRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "Adaboost Regressor": AdaBoostRegressor()
            }
            
            model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models)
            
            best_model_scoer = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_scoer)
            ]
            best_model = models[best_model_name]
            
            if best_model_scoer<0.6:
                raise CustomException("No Best Model Found")
            
            logging.info('Model training completed! Report Generated!')
            
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path, 
                obj=best_model
            )
            
            predicted_output = best_model.predict(x_test)
            r2_score_name = r2_score(y_test, predicted_output)
            return r2_score_name
            
            
        except Exception as e:
            raise CustomException(e, sys)
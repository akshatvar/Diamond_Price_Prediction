import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self,train_data,test_data):
        try:
            logging.info('Spliting dependent and independent variables from test and train dataset')

            X_train,X_test,y_train,y_test = (
                train_data[:,:-1],
                test_data[:,:-1],
                train_data[:,-1],
                test_data[:,-1]
            )
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                'Decision Tree':DecisionTreeRegressor(),
                'Random Forest':RandomForestRegressor(),
                'XGboost':xgb.XGBRegressor()
            }
            model_report:dict = evaluate_model(X_train,X_test,y_train,y_test,models)

            logging.info(f'Model Report : {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'Best Model Name : {best_model_name}, R2 Score:{best_model_score}')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            print(model_report)
            print('\n====================================================================================\n')

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                object=best_model
            )
        except Exception as e:
            logging.info('Error occured at initiate model training')
            raise CustomException(e,sys)




import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from src.componenets.data_ingestion import DataIngestion
from src.componenets.data_tranformation import DataTransformation
from src.componenets.model_trainer import ModelTrainer

if __name__=='__main__':
    obj = DataIngestion()
    train_data_path,test_data_path = obj.initiate_data_ingestion()
    print(train_data_path)
    print(test_data_path)
    data_transformation = DataTransformation()
    train_arr,test_arr,preprocessor_pkl_path = data_transformation.initiate_data_transformation(test_data_path,test_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)

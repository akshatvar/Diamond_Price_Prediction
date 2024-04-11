import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize data ingestion configuration

@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

# Create the data ingestion class

class DataIngestion:
    def __init__(self):
        self.data_ingestion = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data','diamonds.csv'))
            logging.info('Dataset read as pandas dataframe sucessfully')

            os.makedirs(os.path.dirname(self.data_ingestion.raw_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion.raw_data_path,index=False)
            logging.info('Raw data created sucessfully')

            train_data,test_data = train_test_split(df,test_size=0.30,random_state=27)

            train_data.to_csv(self.data_ingestion.train_data_path,index=False,header=False)
            test_data.to_csv(self.data_ingestion.test_data_path,index=False,header=True)
            logging.info('Ingestion of data is completed sucessfully')

            return(
                self.data_ingestion.train_data_path,
                self.data_ingestion.test_data_path
            )
        except Exception as e:
            logging.info('Exception occured at Data Ingestion')
            raise CustomException(e,sys)

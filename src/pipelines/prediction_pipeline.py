import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_file
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            preprocessor_pkl_path = os.path.join('artifacts','preprocessor.pkl')
            model_pkl_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_file(preprocessor_pkl_path)
            model = load_file(model_pkl_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)
            return pred
        except Exception as e:
            logging.info('Error occured at Predict')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 carat:float,
                 depth:float,
                 table:float,
                 x:float,
                 y:float,
                 z:float,
                 cut:str,
                 color:str,
                 clarity:str):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity
    
    def get_data_as_data_frame(self):
        try:
            custom_data = {
                'carat':[self.carat],
                'depth':[self.depth],
                'table':[self.table],
                'x':[self.x],
                'y':[self.y],
                'z':[self.z],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(custom_data)
            logging.info('Dataframe is created from input data')
            return df
        except Exception as e:
            logging.info('Error occured at get data as data frame')
            raise CustomException(e,sys)
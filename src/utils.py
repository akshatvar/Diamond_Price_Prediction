import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.logger import logging
from src.exception import CustomException

def save_object(file_path,object):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(object,file_obj)
    except Exception as e:
        logging.info('Error occured at save object')
        raise CustomException(e,sys)
    
def evaluate_model(X_train,X_test,y_train,y_test,models):
    try:
        report = {}
        for key,model in models.items():
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            score = r2_score(y_test,y_pred)
            report[key] = score
        return report
    except Exception as e:
        logging.info('Error occured at evaluate_model')
        raise CustomException(e,sys)

def load_file(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Error occured at load file')
        raise CustomException(e,sys)
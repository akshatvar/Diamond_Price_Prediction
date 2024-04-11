from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import sys,os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_pkl_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Initiated')

            categorical_col = ['cut','color','clarity']
            numerical_col = ['carat','depth','table','x','y','z']

            # define custom ranking for ordianl variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            logging.info('Pipeline Initiated')

            numerical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ('numerical_pipeline',numerical_pipeline,numerical_col),
                ('categorical_pipeline',categorical_pipeline,categorical_col)
            ])
            logging.info('Pipeline Completed')
            return preprocessor
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info('Read Train and Test data completed')
            logging.info(f'Train Dataframe head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe head: \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessor_obj = self.get_data_transformation_object()

            target_col = 'price'
            drop_col = ['Unnamed: 0',target_col]

            # dependent and independent feature
            input_feature_train_df = train_df.drop(columns=drop_col,axis=1)
            target_feature_train_df = train_df[target_col]

            input_feature_test_df = test_df.drop(columns=drop_col,axis=1)
            target_feature_test_df = test_df[target_col]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training and testing datasets.')

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_tranformation_config.preprocessor_pkl_path,
                object=preprocessor_obj
            )

            logging.info('Preprocessor Pickle is created and saved')

            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_pkl_path
            )
        except Exception as e:
            logging.info('Error occured at initiate data transformation')
            raise CustomException(e,sys)



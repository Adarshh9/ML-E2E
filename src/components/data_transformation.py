import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder ,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts" ,"preprocessor.pickle")

class DataTransformation():
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is resposible for data transformation
        '''
        try:
            numerical_features = ['writing_score' ,'reading_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer' ,SimpleImputer(strategy='median')),# handling missing values
                    ('scaler' ,StandardScaler(with_mean=False))# scaling
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer' ,SimpleImputer(strategy='most_frequent')),# handling missing values
                    ('one_hot_encoder',OneHotEncoder()),# OHE
                    ('scaler' ,StandardScaler(with_mean=False))# scaling
                ]
            )

            logging.info('Numerical features standard scaling completed!')
            logging.info('Categorical features encoding completed!')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline' ,num_pipeline ,numerical_features),
                    ('cat_pipeline' ,cat_pipeline ,categorical_features)
                ]
            )

            logging.info('Preprocessor step completed!')

            return preprocessor

        except Exception as e:
            raise CustomException(e ,sys)
        
    def initiate_data_transformation(self ,train_path ,test_path):
        try:
            train_df = pd.read_csv('artifacts/train.csv')
            test_df = pd.read_csv('artifacts/test.csv')

            logging.info('Read train and test data completed!')
            logging.info('Loading preprocessor object')

            preprocessor_object = self.get_data_transformer_object()

            target_column = 'math_score'
            numerical_columns = ['writing_score' ,'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column] ,axis=1)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column] ,axis=1)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing on train and test dataframe')

            input_feature_train_arr = preprocessor_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_object.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr ,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr ,np.array(target_feature_test_df)]

            logging.info('Saved preprocessing object')

            save_object(
                file_path = self.transformation_config.preprocessor_obj_file_path ,
                obj = preprocessor_object
            )

            return (
                train_arr ,
                test_arr ,
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e ,sys)
"""This is to do the transformation in data"""

import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        """To create all the pickle files"""
        try:
            num_features =  ['reading_score', 'writing_score']
            cat_features = [
                'gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),   #handle the outliers
                    ('scaler', StandardScaler()) # do the standard scaling
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    #to handle missing values
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    
                    #One hot encoder
                    ('one_hot_endcoder', OneHotEncoder()),
                    
                    #Standard Scaling
                    ('scaler', StandardScaler(with_mean=False))
                ]
                
            )
            
            logging.info('numerical columns standard scaling completed')
            logging.info('categorical columns standard scaling completed')
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )
            
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)   
        
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('REad Train and Test Data Completed.')
            logging.info('Obtaining preprocessing object')
            preprocessign_obj = self.get_data_transformer_obj()
            
            target_col_name = 'math_score'
            num_col_names = ['writing_score', 'reading_score']
            
            #for train data
            input_feature_train_df = train_df.drop(columns=[target_col_name], axis = 1)
            target_feature_train_df = train_df[target_col_name]
            
            #for test data
            input_feature_test_df = test_df.drop(columns=[target_col_name], axis = 1)
            target_feature_test_df = test_df[target_col_name]
            
            #loggign
            logging.info(
                f"Applying preprocessing object on train and test df'"
            )
            input_feature_train_arr = preprocessign_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessign_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            
            #save pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file,
                obj = preprocessign_obj
            )
            return (
                train_arr, 
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
            
            
        except Exception as e:
            raise CustomException(e, sys)

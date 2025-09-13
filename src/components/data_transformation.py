import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , 'preprocessor.pkl') #class variable

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()  

    def get_data_transformer_object(self):
        '''
        This function is responsible for data Transformation 
        '''
        try:
            #Numerical features
            numerical_columns = ['writing_score' , 'reading_score'] 

            #Categorial features
            categorical_columns = [ 'gender' , 
                                    'race_ethnicity' , 
                                    'parental_level_of_education',
                                    'lunch' , 
                                    'test_preparation_course'
                                  ]
            
            #to decalre the steps like handling missing values , standard scaling for numerical features
            num_pipeline = Pipeline(
                                    steps = [
                                                ('imputer' , SimpleImputer(strategy='median')), #handle missing values
                                                ('scaler' , StandardScaler())
                                            ]
                                   )
            
            logging.info('Numerical columns standard scaling completed')

            #to decalre the steps like handling missing values , standard scaling for categorical features
            cat_pipeline = Pipeline(
                                    steps = [
                                                ('imputer' , SimpleImputer(strategy='most_frequent')), #handle missing values
                                                ('one_hot_encoder' , OneHotEncoder()), #convert categorial field to numeric
                                                ('scaler' , StandardScaler(with_mean=False))
                                            ]
                                   )
            
            logging.info('Categorical columns encoding completed')
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #combine numerical columns with categorical columns
            preprocessor = ColumnTransformer(
                                             [
                                                 ('num_pipeline' , num_pipeline, numerical_columns),
                                                 ('cat_pipeline' , cat_pipeline , categorical_columns)
                                             ]
                                            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e , sys)
            
    
    def initiate_data_transformation(self , train_path, test_path):
        try:
            train_df = pd.read_csv(train_path) #get the training data from filepath from data ingestion
            test_df = pd.read_csv(test_path)   #get the test data from filepath from data ingestion

            logging.info('Read Train and Test data completed')
            logging.info('Obtaining preprocessor object')

            preprocessing_obj = self.get_data_transformer_object() 

            target_column_name  = 'math_score' #variable
            numerical_columns = ['writing_score' , 'reading_score'] #list

            input_feature_train_df = train_df.drop(columns = target_column_name , axis=1)  #X columns
            target_feature_train_df = train_df[target_column_name] #Y columns

            input_feature_test_df = test_df.drop(columns = target_column_name , axis=1)  #X columns
            target_feature_test_df = test_df[target_column_name] #Y columns

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            
            #model fitting on train and test data // pickle file
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                                input_feature_train_arr , np.array(target_feature_train_df)
                             ]
            
            test_arr = np.c_[
                                input_feature_test_arr , np.array(target_feature_test_df)
                             ] 

            logging.info(f"saved preprocessing object.")    
            
            #save the pickle file
            save_object(   
                        file_path = self.data_transformation_config.preprocessor_obj_file_path,
                        obj = preprocessing_obj
                       )  ##written in Utils

            return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path
                   )       
        except Exception as e:
            raise CustomException(e, sys)
import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline=Pipeline(
                steps=[
                    ("imputing",SimpleImputer(strategy="median")),
                    ("scaling",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputing",SimpleImputer(strategy="most_frequent")),
                    ("Onehot_encoding",OneHotEncoder())
                ]
            )
            logging.info("categorical info encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("read the data from train and test data paths as DF")

            preprocessor_obj=self.get_data_transformer_obj()
            
            target_column="math_score"
            numerical_columns = ["writing_score", "reading_score"]
            input_feature_train=train_df.drop([target_column],axis=1)
            target_feature_train=train_df[target_column]

            input_feature_test=test_df.drop([target_column],axis=1)
            target_feature_test=test_df[target_column]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test)
            ]

            logging.info(f"Saved preprocessing object.")

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)

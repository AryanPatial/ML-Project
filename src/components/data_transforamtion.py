import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import pickle

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Mainly responsible for data transformation for numerical and categorical features.
        Creates a preprocessing pipeline for numerical and categorical features.
        """
        try:
            # Example: You’ll need to adjust column names for your dataset
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical pipeline created")
            logging.info("Categorical pipeline created")

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_features),
                    ("cat", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and Test data loaded for transformation")


            # Split into input features (X) and target column (y)
            target_column = "math_score"   # change this based on your dataset
            numerical_columns = ["writing_score", "reading_score"]
            
            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            logging.info("Splitting into input and target done")

            logging.info("Obtaining preprocessing object")

            # Get preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Apply transformation
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            logging.info("Preprocessing applied on train and test data")

            # Save the preprocessor object to artifacts

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessor object saved")

            # Return transformed arrays along with target
            return (
                np.c_[X_train_transformed, np.array(y_train)],
                np.c_[X_test_transformed, np.array(y_test)],
                self.transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)



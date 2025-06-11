import os
RAW_DIR = "artifacts/raw"
TRAIN_PATH = os.path.join(RAW_DIR,'customers_train.csv')
TEST_PATH = os.path.join(RAW_DIR,'customers_test.csv')
DATA_PATH=os.path.join(RAW_DIR,'customers_data.csv')


PROCESSED_DIR = "artifacts/processed"

####################### MODEL TRAINING #################
MODEL_OUTPUT_PATH = "artifacts/models/random_forest_model.pkl"
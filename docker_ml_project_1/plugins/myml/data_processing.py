import pandas as pd
from sklearn.model_selection import train_test_split
# from src.feature_store import RedisFeatureStore
from myml.logger import get_logger
# from src.custom_exception import CustomException
#from config.paths_config import *

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, df): #, feature_store : RedisFeatureStore):
        self.data=df
        #self.feature_store = feature_store
        logger.info("Your Data Processing is intialized...")
   
    def preprocess_data(self):
        try:

            self.data['TotalCharges'] = pd.to_numeric(self.data['TotalCharges'], errors='coerce')
            self.data.dropna(subset=['TotalCharges'], inplace=True)
            self.data['TotalCharges'] = self.data['TotalCharges'].fillna(2279) # 2279 mean value in data
            self.data['TotalCharges'] = self.data['TotalCharges'].astype(str)
            self.data['TotalCharges'] = self.data['TotalCharges'].str.replace(' ','2279') # remove space string in data
            self.data['TotalCharges'] = self.data['TotalCharges'].astype(float)
            self.data['PhoneService'].fillna('No')
            self.data['tenure'] = self.data['tenure'].fillna(self.data['tenure'].mean())
            self.data['Contract'] = self.data['Contract'].dropna()
            self.data['PhoneService'] = self.data['PhoneService'].map({'Yes':1,'No':0})
            df_processed = self.data.join(pd.get_dummies(self.data['Contract']).astype(int))

            logger.info("Data Preprocessing done...")

        except Exception as e:
            logger.error(f"Error while preprocessing data {e}")
            #raise CustomException(str(e))    
        return df_processed

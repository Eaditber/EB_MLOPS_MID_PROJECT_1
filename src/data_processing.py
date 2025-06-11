import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from src.feature_store import RedisFeatureStore
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *

logger = get_logger(__name__)


class DataProcessing:
    def __init__(self, train_data_path , test_data_path , feature_store : RedisFeatureStore):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.data=None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train=None
        self.y_test = None

        self.X_resampled = None
        self.y_resampled = None

        self.feature_store = feature_store
        logger.info("Your Data Processing is intialized...")
    
    def load_data(self):
        try:
            self.data = pd.read_csv(self.train_data_path)
            self.test_data = pd.read_csv(self.test_data_path)
            logger.info("Read the data sucesfully")
        except Exception as e:
            logger.error(f"Error while reading data {e}")
            raise CustomException(str(e))
    
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
            self.data =self.data.join(pd.get_dummies(self.data['Contract']).astype(int))
            
            logger.info("Data Preprocessing done...")

        except Exception as e:
            logger.error(f"Error while preprocessing data {e}")
            raise CustomException(str(e))
    
    def handle_imbalance_data(self):
        try:
            X = self.data[['TotalCharges','Month-to-month','One year','Two year','PhoneService','tenure']]
            y = self.data['Churn']

            smote = SMOTE(random_state=42)
            self.X_resampled, self.y_resampled = smote.fit_resample(X, y)

            logger.info("Hanled imbalance data sucesfully...")

        except Exception as e:
            logger.error(f"Error while imabalanced handling data {e}")
            raise CustomException(str(e))
    
    def store_feature_in_redis(self):
        try:
            batch_data = {}
            for idx,row in self.data.iterrows():
                entity_id = row["customerID"]
                features = {
                    "TotalCharges": row['TotalCharges'],
                    "Month-to-month": row['Month-to-month'],
                    "One year": row['One year'],
                    "Two year": row['Two year'],
                    "PhoneService" : row['PhoneService'],
                    "tenure" : row['tenure'],
                    "Churn" : row['Churn']
                    
                }
                batch_data[entity_id] = features
            self.feature_store.store_batch_features(batch_data)
            logger.info("Data has been feeded into Feature Store..")
        except Exception as e:
            logger.error(f"Error while feature storing data {e}")
            raise CustomException(str(e))
        
    def retrive_feature_redis_store(self,entity_id):
        features = self.feature_store.get_features(entity_id)
        if features:
            return features
        return None
    
    def run(self):
        try:
            logger.info("Starting our Data Processing Pipleine...")
            self.load_data()
            self.preprocess_data()
            self.handle_imbalance_data()
            self.store_feature_in_redis()

            logger.info("End of pipeline Data Processing...")

        except Exception as e:
            logger.error(f"Error while Data Processing Pipleine {e}")
            raise CustomException(str(e))
        
if __name__=="__main__":
    feature_store = RedisFeatureStore()

    data_processor = DataProcessing(TRAIN_PATH,TEST_PATH,feature_store)
    data_processor.run()

    print(data_processor.retrive_feature_redis_store(entity_id='3668-QPYBK'))
        



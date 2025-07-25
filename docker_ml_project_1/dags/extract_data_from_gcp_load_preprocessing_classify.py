from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator

from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy
from airflow.timetables.interval import Timetable  # You might need this import if using complex schedules
from datetime import timedelta
import pendulum
import joblib
import numpy as np
#from config.paths_config import C

from myml.data_processing import DataProcessing
# This DAG extracts data from Google Cloud Storage, processes it, and loads it into a PostgreSQL database.

#### TRANSFORM STEP....
def load_to_preprocessing_classification(file_path, tab_name_raw, tab_name_processed, model_output_path):
    conn = BaseHook.get_connection('postgres_default')
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@docker_ml_project_1-postgres-1:{conn.port}/{conn.schema}")
    df = pd.read_csv(file_path)
    df['insert_date'] = datetime.now()

    df.to_sql(name=tab_name_raw, con=engine, if_exists="append", index=False) # name="customers_data" table name
    # Get mean of a column from the processed table
    
    with engine.connect() as conn_sql:
        try:
            df_db = pd.read_sql(f"SELECT avg(tenure) as avg_tenure FROM {tab_name_processed}", conn_sql)
            mean_tenure = df_db['avg_tenure'][0]
        except Exception as e:
            mean_tenure = df['tenure'].mean()
        # Save mean_tenure to JSON file
        import json
        with open('plugins/myml/mean_tenure.json', 'w') as f:
            json.dump({'mean_tenure': mean_tenure}, f)
        data_processor = DataProcessing(df, mean_tenure)  # Pass mean_tenure to DataProcessing
        df_processed = data_processor.preprocess_data()
        # --- Model Loading (Add error handling for robustness) ---

    try:
        loaded_model = joblib.load(model_output_path)
        print(f"Model loaded successfully from {model_output_path}")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_output_path}. Please check the path.")
    except Exception as e:
        print(f"ERROR: An error occurred while loading the model: {e}")
    features=['TotalCharges','Month-to-month','One year','Two year','PhoneService','tenure']
    df_processed['prediction']= loaded_model.predict(df_processed[features])
    df_processed['mean_tenure']=mean_tenure
    df_processed['insert_date'] = datetime.now()
    df_processed.to_sql(name=tab_name_processed, con=engine, if_exists="append", index=False)


# Define the DAG
with DAG(
    dag_id="extract_customers_data_preprocessing_classification",
    #schedule=timedelta(days=1), # Changed 'schedule_interval' to 'schedule'
    #start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    start_date=pendulum.datetime(2023, 1, 1, 0, 0, 0, tz="Asia/Jerusalem"), # 12:00 AM Israel time on Jan 1, 2023
    schedule="0 12 * * *", # CRON expression for 12:00 PM daily
    catchup=False,
) as dag:

    # Extract STEP...
    list_files = GCSListObjectsOperator(
        task_id="list_files",
        bucket="eb_mlops_bucket", 
    )

    download_file = GCSToLocalFilesystemOperator(
        task_id="download_file",
        bucket="eb_mlops_bucket", 
        object_name="original_dataset.csv", 
        filename="/tmp/original_dataset.csv", 
    )
    
    ### TRANSFORM AND LOAD....
    load_preprocessing_classification = PythonOperator(
        task_id="load_to_and_preprocessing",
        python_callable=load_to_preprocessing_classification,
        op_kwargs={"file_path": "/tmp/original_dataset.csv", 
                   "tab_name_raw": "fact_customers_data_daily",
                   "tab_name_processed": "fact_customers_data_daily_processed_w_predictions",
                   "model_output_path": "plugins/myml/random_forest_model.pkl"}
    )
#DAGGGGG
    list_files >> download_file >> load_preprocessing_classification

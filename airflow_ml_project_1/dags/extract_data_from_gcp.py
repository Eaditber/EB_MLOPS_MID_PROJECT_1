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

#### TRANSFORM STEP....
def load_to_sql(file_path):
    conn = BaseHook.get_connection('postgres_default')  
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@airflow_ml_project_1-postgres-1:{conn.port}/{conn.schema}")

    df = pd.read_csv(file_path)
    df.to_sql(name="fact_customers_data", con=engine, if_exists="replace", index=False) # name="customers_data" table name

# Define the DAG
with DAG(
    dag_id="extract_customers_data",
    schedule=timedelta(days=1), # Changed 'schedule_interval' to 'schedule'
    start_date=datetime(2023, 1, 1),
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
        object_name="database_input.csv", 
        filename="/tmp/database_input.csv", 
    )
    
    ### TRANSFORM AND LOAD....
    load_data = PythonOperator(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={"file_path": "/tmp/database_input.csv"}
    )
#DAGGGGG
    list_files >> download_file >> load_data

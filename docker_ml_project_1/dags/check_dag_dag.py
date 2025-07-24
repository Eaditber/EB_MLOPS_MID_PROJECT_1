from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.utils.state import DagRunState # <-- Import DagRunState

with DAG(
    dag_id='dag_n',
    start_date=datetime(2023, 1, 1),
    schedule=timedelta(days=1),
    catchup=False,
    tags=['example'],
) as dag_n:
    wait_for_dag_preprocess  = ExternalTaskSensor(
        task_id='wait_for_dag_preproc_completion',
        external_dag_id='extract_customers_data_preprocessing',
        # external_task_id=None, # Explicitly stating this for clarity, it's None by default
        
        # Change 'success' to DagRunState.SUCCESS
        allowed_states=[DagRunState.SUCCESS], 
        
        # Change 'failed' and 'skipped' to DagRunState.FAILED and DagRunState.SKIPPED
        failed_states=[DagRunState.FAILED], 
        
        poke_interval=60,
        timeout=60 * 60 * 24,
    )

    task_in_n = BashOperator(
        task_id='start_task_in_n_after_a',
        bash_command='echo "DAG N is now running after DAG A completed!"',
    )

    wait_for_dag_preprocess >> task_in_n
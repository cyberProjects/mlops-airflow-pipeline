from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.operators.lambda_function import LambdaInvokeFunctionOperator

from jobs.launch_credit_xgb_training_job import launch_training_job
from jobs.evaluate_credit_xgb_model_job import evaluate_model_performance

default_args = {
    'owner': 'msecindustries',
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

dag = DAG(
    'credit_default_pipeline',
    default_args=default_args,
    description='Real end-to-end MLOps pipeline for credit default prediction',
    schedule_interval=None,
    catchup=False,
    tags=['ml', 'aws', 'pipeline'],
)

def check_lambda_connection():
    import boto3

    try:
        client = boto3.client('lambda', region_name='us-east-1')
        response = client.list_functions()
        function_names = [fn['FunctionName'] for fn in response.get('Functions', [])]
        print(f"Successfully connected to AWS Lambda. Found {len(function_names)} functions: {function_names}")
    except Exception as e:
        print(f"Failed to connect to AWS Lambda: {str(e)}")
        raise

# Task Definitions
check_aws_task = PythonOperator(
    task_id='check_lambda_connection',
    python_callable=check_lambda_connection,
    dag=dag,
)

clean_s3_task = LambdaInvokeFunctionOperator(
    task_id='invoke_clean_s3_raw',
    function_name='msecindsutries-ingest-clean-s3-raw',
    invocation_type='RequestResponse',
    aws_conn_id='aws_default',
    log_type='Tail',
    dag=dag,
)

clean_rds_task = LambdaInvokeFunctionOperator(
    task_id='invoke_clean_rds',
    function_name='msecindustries-ingest-clean-rds',
    invocation_type='RequestResponse',
    aws_conn_id='aws_default',
    log_type='Tail',
    dag=dag,
)

merge_task = LambdaInvokeFunctionOperator(
    task_id='invoke_merge_cleaned_data',
    function_name='msecindustries-merge-prep-train-data',
    invocation_type='RequestResponse',
    aws_conn_id='aws_default',
    log_type='Tail',
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='start_sagemaker_training',
    python_callable=launch_training_job,
    dag=dag,
)

evaluate_model_task = PythonOperator(
    task_id='evaluate_model_performance',
    python_callable=evaluate_model_performance,
    provide_context=True,
    dag=dag,
)

# Task Dependency Chain
check_aws_task >> clean_s3_task >> clean_rds_task >> merge_task >> train_model_task >> evaluate_model_task
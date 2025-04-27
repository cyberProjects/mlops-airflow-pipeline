import boto3
import sagemaker
from sagemaker.estimator import Estimator
import time
from datetime import datetime

def wait_for_training_job(sagemaker_client, training_job_name):
    while True:
        response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        status = response['TrainingJobStatus']
        print(f"Training job {training_job_name} status: {status}")
        
        if status == 'Completed':
            print(f"Training job {training_job_name} completed successfully.")
            break
        elif status in ['Failed', 'Stopped']:
            raise Exception(f"Training job {training_job_name} ended with status: {status}")
        else:
            time.sleep(30)

def launch_training_job():
    # Configuration
    region = 'us-east-1'
    s3_input_data = 's3://msecindustries-training/final-dataset/credit_default_final_training.csv'
    s3_output_model = 's3://msecindustries-models/xgboost-model-artifacts/'

    # SageMaker session and role
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)
    sm_client = boto_session.client('sagemaker')

    role = ''

    # Built-in XGBoost container URI
    container_uri = sagemaker.image_uris.retrieve(
        framework='xgboost',
        region=region,
        version='1.5-1'
    )

    # Create unique job name
    job_name = f"credit-default-xgboost-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Define the estimator
    xgb_estimator = Estimator(
        image_uri=container_uri,
        role=role,
        instance_count=1,
        instance_type='ml.m5.xlarge',
        output_path=s3_output_model,
        sagemaker_session=sagemaker_session
    )

    # Set hyperparameters
    xgb_estimator.set_hyperparameters(
        objective='binary:logistic',
        num_round=100,
        max_depth=5,
        eta=0.2,
        gamma=4,
        min_child_weight=6,
        subsample=0.8
    )

    # Input format (SageMaker expects "train" channel)
    train_input = sagemaker.inputs.TrainingInput(
        s3_data=s3_input_data,
        content_type='text/csv'
    )

    # Launch the training job
    print(f"Starting training job: {job_name}")
    xgb_estimator.fit({'train': train_input}, job_name=job_name)

    # Wait for training job to complete
    wait_for_training_job(sm_client, job_name)

    print(f"Training job {job_name} finished successfully.")
    return {"TrainingJobName": job_name}
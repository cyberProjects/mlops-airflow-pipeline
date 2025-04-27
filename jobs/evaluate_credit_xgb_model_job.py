import boto3
import pandas as pd
import os
import tarfile
import xgboost as xgb
from sklearn.metrics import accuracy_score

def evaluate_model_performance(**kwargs):
    # Configuration
    region = 'us-east-1'
    bucket_name = 'msecindustries-models'

    # Setup boto3 client
    s3 = boto3.client('s3', region_name=region)

    # Pull the job_name passed via Airflow XComs
    job_name = kwargs['ti'].xcom_pull(task_ids='start_sagemaker_training')['TrainingJobName']

    # Download model artifact
    print("Downloading model artifact...")
    model_local_path = '/tmp/model.tar.gz'
    s3_model_key = f"xgboost-model-artifacts/{job_name}/output/model.tar.gz"
    s3.download_file(bucket_name, s3_model_key, model_local_path)

    # Extract model.tar.gz
    print("Extracting model...")
    with tarfile.open(model_local_path, 'r:gz') as tar:
        tar.extractall(path='/tmp/')

    # Find extracted model file
    extracted_files = os.listdir('/tmp/')
    if 'xgboost-model' not in extracted_files:
        raise ValueError("No xgboost-model file found after extraction!")

    model_file_path = os.path.join('/tmp/', 'xgboost-model')

    # Load the model
    print(f"Loading model from {model_file_path}...")
    booster = xgb.Booster()
    booster.load_model(model_file_path)

    # Download validation set
    print("Downloading validation data...")
    validation_local_path = '/tmp/validation.csv'
    s3.download_file('msecindustries-training', 'final-dataset/credit_default_final_training.csv', validation_local_path)

    # Load validation data
    print("Loading validation dataset...")
    val_df = pd.read_csv(validation_local_path)

    X_val = val_df.drop(columns=['default_payment_next_month'])
    y_val = val_df['default_payment_next_month']

    # Prepare validation set as DMatrix
    dval = xgb.DMatrix(X_val)

    # Predict
    print("Predicting validation set...")
    preds = booster.predict(dval)

    # Calculate accuracy
    acc = accuracy_score(y_val, preds.round())
    print(f"Validation Accuracy: {acc:.4f}")

    # Threshold check
    threshold = 0.80
    if acc < threshold:
        raise ValueError(f"Model accuracy {acc:.4f} is below acceptable threshold of {threshold}!")

    print("Model passed quality gate.")
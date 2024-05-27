# src/main.py

import os
from pathlib import Path
import logging
import pandas as pd
import joblib
import yaml

from aws_utils import download_refined_data, upload_artifacts
from train_model import split_data, train_model
from model_score import score_model
from model_evaluation import evaluate_model

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
INPUT_BUCKET_NAME = os.getenv("INPUT_BUCKET_NAME", "wqj1453-cloud-project")
OUTPUT_BUCKET_NAME = os.getenv("OUTPUT_BUCKET_NAME", "wqj1453-artifacts-project")
ARTIFACTS_PREFIX = os.getenv("ARTIFACTS_PREFIX", "artifacts/")
CONFIG_REF = os.getenv("CONFIG_REF", "config/initial-config.yaml")

def load_config(config_ref: str) -> dict:
    config_file = Path(config_ref)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file at {config_file.absolute()} does not exist")

    with config_file.open() as f:
        return yaml.load(f, Loader=yaml.SafeLoader)

def main():
    # Load the configuration
    config = load_config(CONFIG_REF)
    run_config = config.get("run_config", {})
    version = run_config.get("version", "default")

    # Define paths
    local_refined_data_path = Path('output_runs/refined_data.csv')
    artifacts_path = Path('output_runs/')
    model_save_path = artifacts_path / 'trained_model.pkl'
    model_scoring_output_dir = artifacts_path
    model_evaluation_output_path = artifacts_path / 'model_evaluation.txt'

    # Step 1: Download refined data from S3
    logger.info("Starting data download from S3...")
    download_refined_data(INPUT_BUCKET_NAME, 'path/to/refined_data.csv', local_refined_data_path)

    # Load the refined data
    refined_data = pd.read_csv(local_refined_data_path)

    # Separate features and target variable
    x_data = refined_data.drop('target', axis=1)  # Adjust column name as needed
    y_data = refined_data['target']

    # Step 2: Split the data
    x_train, x_val, y_train, y_val = split_data(x_data, y_data)

    # Step 3: Train the model
    logger.info("Starting model training...")
    model = train_model(x_train, y_train, x_val, y_val, artifacts_path)

    # Step 4: Score the model
    logger.info("Starting model scoring...")
    x_val_flat = x_val.values.reshape(x_val.shape[0], -1)  # Flatten the validation set
    model_scoring = score_model(model, x_val_flat, model_scoring_output_dir)

    # Step 5: Evaluate the model
    logger.info("Starting model evaluation...")
    evaluate_model(model, x_val, y_val, model_evaluation_output_path)

    # Step 6: Upload artifacts to S3
    logger.info("Uploading artifacts to S3...")
    upload_artifacts(artifacts_path, OUTPUT_BUCKET_NAME, ARTIFACTS_PREFIX + version + "/")

    logger.info("Pipeline execution completed successfully.")

if __name__ == '__main__':
    print("Starting pipeline...")
    main()

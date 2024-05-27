import argparse
import datetime
import logging.config
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
from src import aws_utils as aws
from src import train_model as tm
from src import model_evaluation as me
from src import model_score as ms

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("pipeline")

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
            logger.info("Configuration file loaded from %s", config_path)
            return config
        except yaml.YAMLError as e:
            logger.error("Error while loading configuration from %s: %s", config_path, e)
            sys.exit(1)

def setup_artifacts_directory(output_path):
    """Set up the directory for saving artifacts."""
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(output_path) / str(now)
    artifacts.mkdir(parents=True)
    return artifacts

def download_data(bucket_name, s3_key, local_path):
    """Download refined data from S3."""
    try:
        logger.info("Downloading refined data from S3...")
        aws.download_refined_data(bucket_name, s3_key, local_path)
        logger.info("Refined data downloaded successfully.")
    except (FileNotFoundError, PermissionError, RuntimeError) as e:
        logger.error("Failed to download refined data from S3: %s", e)
        sys.exit(1)

def load_data(data_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError as e:
        logger.error("Failed to find the refined data file: %s", e)
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        logger.error("No data: %s", e)
        sys.exit(1)
    except pd.errors.ParserError as e:
        logger.error("Parsing error: %s", e)
        sys.exit(1)

def preprocess_data(data, target_column):
    """Preprocess the data by splitting features and target."""
    x_data = data.drop(columns=target_column)
    y_data = data[target_column]
    return x_data, y_data

def split_and_train_model(x_data, y_data, config, artifacts):
    """Split the data, train the model, and save it."""
    x_train, x_val, y_train, y_val = tm.split_data(x_data, y_data)
    if x_train is None:
        logger.error("Data splitting failed. Exiting pipeline.")
        sys.exit(1)

    model_save_path = config.get("model_training", {}).get("model_save_path", artifacts / "model.pkl")
    model = tm.train_model(x_train, y_train, x_val, y_val, model_save_path)
    if model is None:
        logger.error("Model training failed. Exiting pipeline.")
        sys.exit(1)

    return model, x_val, y_val

def score_and_evaluate_model(model, x_val, y_val, artifacts):
    """Score and evaluate the model."""
    x_val_flat = x_val.values.reshape(x_val.shape[0], -1)
    scoring_results = ms.score_model(model, x_val_flat, artifacts)
    if scoring_results is None:
        logger.error("Model scoring failed. Exiting pipeline.")
        sys.exit(1)

    evaluation_results = me.evaluate_model(model, x_val, y_val, artifacts)
    if evaluation_results is None:
        logger.error("Model evaluation failed. Exiting pipeline.")
        sys.exit(1)

def upload_artifacts_if_needed(artifacts, aws_config):
    """Upload artifacts to S3 if configured to do so."""
    if aws_config.get("upload", False):
        aws.upload_artifacts(artifacts, aws_config["artifacts_bucket_name"], "artifacts")

def main(config_path):
    """Run the end-to-end pipeline."""
    config = load_config(config_path)
    aws_config = config.get("aws", {})
    aws_config["data_bucket_name"] = os.getenv("S3_DATA_BUCKET_NAME", aws_config.get("data_bucket_name"))
    aws_config["artifacts_bucket_name"] = os.getenv("S3_ARTIFACTS_BUCKET_NAME", aws_config.get("artifacts_bucket_name"))
    aws_config["aws_profile"] = os.getenv("AWS_PROFILE", aws_config.get("aws_profile"))

    run_config = config.get("run_config", {})
    artifacts = setup_artifacts_directory(run_config.get("output", "output_runs"))

    refined_data_path = artifacts / "refined_data.csv"
    download_data(aws_config["data_bucket_name"], run_config["data_s3_key"], refined_data_path)

    data = load_data(refined_data_path)
    x_data, y_data = preprocess_data(data, run_config["target_column"])

    model, x_val, y_val = split_and_train_model(x_data, y_data, config, artifacts)

    score_and_evaluate_model(model, x_val, y_val, artifacts)

    upload_artifacts_if_needed(artifacts, aws_config)

    logger.info("Pipeline execution completed. All outputs saved in: %s", artifacts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline.")
    parser.add_argument("--config", default="config/initial-config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)

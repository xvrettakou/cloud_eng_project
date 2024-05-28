import argparse
import datetime
import logging.config
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import aws_utils as aws
import train_model as tm
import model_evaluation as me
import model_score as ms

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("pipeline")

def preprocess_data(data, target_column):
    """Preprocess the data by splitting the 'pixels' column and extracting the target column."""
    x_data = data["pixels"].apply(lambda x: [int(pixel) for pixel in x.split()])
    x_data = pd.DataFrame(x_data.tolist())
    y_data = data[target_column]
    logger.info("Features shape: %s, Target shape: %s", x_data.shape, y_data.shape)
    return x_data, y_data

def download_data(bucket_name, s3_key, local_path):
    """Download the refined data from S3."""
    try:
        aws.download_refined_data(bucket_name, s3_key, local_path)
        logger.info("Refined data downloaded successfully.")
    except FileNotFoundError as e:
        logger.error("Failed to download refined data from S3: File not found - %s", e)
        sys.exit(1)

def split_and_train_model(x_data, y_data, artifacts):
    """Split the data into training and validation sets, then train the model."""
    x_train, x_val, y_train, y_val = tm.split_data(x_data, y_data)
    if x_train is None:
        logger.error("Data splitting failed. Exiting pipeline.")
        sys.exit(1)
    logger.info("x_train shape: %s", x_train.shape)
    logger.info("y_train shape: %s", y_train.shape)
    logger.info("x_val shape: %s", x_val.shape)
    logger.info("y_val shape: %s", y_val.shape)
    model_save_path = artifacts / "trained_model.pkl"
    model = tm.train_model(x_train, y_train, x_val, y_val, model_save_path)
    if model is None:
        logger.error("Model training failed. Exiting pipeline.")
        sys.exit(1)
    return model, x_val, y_val

def score_and_evaluate_model(model, x_val, y_val, artifacts):
    """Score the model, evaluate its performance, and save the results."""
    try:
        x_val_flat = x_val.values.reshape(x_val.shape[0], -1)
        scoring_results = ms.score_model(model, x_val_flat, artifacts)
        if scoring_results is None:
            logger.error("Model scoring failed. Exiting pipeline.")
            sys.exit(1)
        val_predictions = scoring_results["predictions"]  # Get predictions from scoring results
        evaluation_results_path = artifacts / "evaluation_results.txt"
        accuracy = me.evaluate_model(y_val, val_predictions, evaluation_results_path)
        if accuracy is None:
            logger.error("Model evaluation failed. Exiting pipeline.")
            sys.exit(1)

    except ValueError as e:
        logger.error("Value error during model evaluation: %s", e)
        sys.exit(1)
    except OSError as e:
        logger.error("OS error during model evaluation: %s", e)
        sys.exit(1)

def upload_artifacts_if_needed(artifacts, aws_config):
    """Upload artifacts to S3 if specified in the configuration."""
    if aws_config.get("upload", False):
        aws.upload_artifacts(artifacts, aws_config["artifacts_bucket_name"], "artifacts")

def main(config_path):
    """ Load configuration file for parameters and run config """
    with open(config_path, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError:
            logger.error("Error while loading configuration from %s", config_path)
            sys.exit(1)
        else:
            logger.info("Configuration file loaded from %s", config_path)

    # Override AWS configuration with environment variables
    aws_config = config.get("aws", {})
    aws_config["data_bucket_name"] = os.getenv("S3_DATA_BUCKET_NAME", aws_config.get("data_bucket_name"))
    aws_config["artifacts_bucket_name"] = os.getenv("S3_ARTIFACTS_BUCKET_NAME", aws_config.get("artifacts_bucket_name"))
    #aws_config["aws_profile"] = os.getenv("AWS_PROFILE", aws_config.get("aws_profile"))

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "output_runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Download refined data from S3
    refined_data_path = artifacts / "refined_data.csv"
    download_data(aws_config["data_bucket_name"], run_config["data_s3_key"], refined_data_path)

    # Load the data
    try:
        data = pd.read_csv(refined_data_path)
        logger.info("Columns in the DataFrame: %s", data.columns)
    except FileNotFoundError as e:
        logger.error("Failed to find the refined data file: %s", e)
        sys.exit(1)

    x_data, y_data = preprocess_data(data, run_config["target_column"])

    model, x_val, y_val = split_and_train_model(x_data, y_data, artifacts)
    score_and_evaluate_model(model, x_val, y_val, artifacts)
    upload_artifacts_if_needed(artifacts, aws_config)

    logger.info("Pipeline execution completed. All outputs saved in: %s", artifacts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline.")
    parser.add_argument("--config", default="config/initial-config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)

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

def main(config_path):
    # Load configuration file for parameters and run config
    with open(config_path, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            logger.error("Error while loading configuration from %s", config_path)
            sys.exit(1)
        else:
            logger.info("Configuration file loaded from %s", config_path)

    # Override AWS configuration with environment variables
    aws_config = config.get("aws", {})
    aws_config["data_bucket_name"] = os.getenv("S3_DATA_BUCKET_NAME", aws_config.get("data_bucket_name"))
    aws_config["artifacts_bucket_name"] = os.getenv("S3_ARTIFACTS_BUCKET_NAME", aws_config.get("artifacts_bucket_name"))
    aws_config["aws_profile"] = os.getenv("AWS_PROFILE", aws_config.get("aws_profile"))

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "output_runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Download refined data from S3
    refined_data_path = artifacts / "refined_data.csv"
    try:
        logger.info("Downloading refined data from S3...")
        aws.download_refined_data(aws_config["data_bucket_name"], run_config["data_s3_key"], refined_data_path)
        logger.info("Refined data downloaded successfully.")
    except Exception as e:
        logger.error("Failed to download refined data from S3: %s", e)
        sys.exit(1)

    # Load the data
    try:
        data = pd.read_csv(refined_data_path)
    except FileNotFoundError as e:
        logger.error("Failed to find the refined data file: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.error("Failed to load the refined data: %s", e)
        sys.exit(1)

    x_data = data.drop(columns=run_config["target_column"])
    y_data = data[run_config["target_column"]]

    # Split the data
    x_train, x_val, y_train, y_val = tm.split_data(x_data, y_data)
    if x_train is None:
        logger.error("Data splitting failed. Exiting pipeline.")
        sys.exit(1)

    # Train the model
    model_save_path = config.get("model_training", {}).get("model_save_path", artifacts / "model.pkl")
    model = tm.train_model(x_train, y_train, x_val, y_val, model_save_path)
    if model is None:
        logger.error("Model training failed. Exiting pipeline.")
        sys.exit(1)

    # Score the model
    x_val_flat = x_val.values.reshape(x_val.shape[0], -1)
    scoring_results = ms.score_model(model, x_val_flat, artifacts)
    if scoring_results is None:
        logger.error("Model scoring failed. Exiting pipeline.")
        sys.exit(1)

    # Evaluate the model
    evaluation_results = me.evaluate_model(model, x_val, y_val, artifacts)
    if evaluation_results is None:
        logger.error("Model evaluation failed. Exiting pipeline.")
        sys.exit(1)

    # Upload artifacts to S3
    if aws_config.get("upload", False):
        aws.upload_artifacts(artifacts, aws_config["artifacts_bucket_name"], "artifacts")

    logger.info("Pipeline execution completed. All outputs saved in: %s", artifacts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the end-to-end pipeline.")
    parser.add_argument("--config", default="config/initial-config.yaml", help="Path to configuration file")
    args = parser.parse_args()
    main(args.config)

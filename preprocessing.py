import argparse
import datetime
import logging.config
from pathlib import Path

import yaml


import src.aws_utils as aws
import src.training_preprocessing as tp

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Acquire, clean, and create features from clouds data"
    )
    parser.add_argument(
        "--config", default="config/initial-config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    # Load configuration file for parameters and run config
    with open(args.config, "r") as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.error.YAMLError as e:
            logger.error("Error while loading configuration from %s", args.config)
        else:
            logger.info("Configuration file loaded from %s", args.config)

    run_config = config.get("run_config", {})

    # Set up output directory for saving artifacts
    now:int = int(datetime.datetime.now().timestamp())
    artifacts = Path(run_config.get("output", "runs")) / str(now)
    artifacts.mkdir(parents=True)

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Acquire data from online repository and save to disk
    ad.acquire_data(run_config["data_source"], artifacts / "clouds.data")
    logger.info("Data acquired and saved to %s", artifacts / "clouds.data")

    # Create structured dataset from raw data; save to disk
    data = cd.create_dataset(artifacts / "clouds.data", config["create_dataset"])
    cd.save_dataset(data, artifacts / "clouds.csv")
    logger.info("Dataset created and saved to %s", artifacts / "clouds.csv")

    # Enrich dataset with features for model training; save to disk
    features = gf.generate_features(data, config["generate_features"])
    logger.info("Features generated")

    # Generate statistics and visualizations for summarizing the data; save to disk
    figures = artifacts / "figures"
    figures.mkdir()
    eda.save_figures(features, figures)
    logger.info("Analytical figures created and saved to %s", artifacts / "figures")


    # Split data into train/test set and train model based on config; save each to disk
    tmo, train, test = tm.train_model(features, config["model"])
    tm.save_data(train, test, artifacts)
    tm.save_model(tmo, artifacts / "trained_model_object.pkl")
    logger.info("Model trained and saved to %s", artifacts / "trained_model_object.pkl")


    # Score model on test set; save scores to disk
    scores = sm.score_model(test, tmo, config["model"])
    sm.save_scores(scores, artifacts / "scores.csv")
    logger.info("Test scores saved to %s", artifacts / "scores.csv")

    # Evaluate model performance metrics; save metrics to disk
    metrics = ep.evaluate_performance(scores, config["metrics"])
    ep.save_metrics(metrics, artifacts / "metrics.yaml")
    logger.info("Test metrics saved to %s", artifacts / "metrics.yaml")

    # Upload all artifacts to S3
    aws_config = config.get("aws")
    if aws_config.get("upload", True):
        aws.upload_artifacts(artifacts, aws_config, now)
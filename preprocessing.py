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


    artifacts = Path("data")

    # Save config file to artifacts directory for traceability
    with (artifacts / "config.yaml").open("w") as f:
        yaml.dump(config, f)

    # Acquire data from s3
    tp.acquire_data(run_config["data_source"], artifacts / "clouds.data")
    logger.info("Data acquired and saved to %s", artifacts / "clouds.data")

    
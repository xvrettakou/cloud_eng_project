print("imports...")
import os
import re
from pathlib import Path

import botocore
import joblib
import yaml
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import aws_utils as aws

artifacts = Path() / "artifacts"
artifacts.mkdir(exist_ok=True)
print(artifacts.absolute())

BUCKET_NAME = os.getenv("BUCKET_NAME", "smf2659-iris")
ARTIFACTS_PREFIX = os.getenv("ARTIFACTS_PREFIX", "artifacts/")
CONFIG_REF = os.getenv("CONFIG_REF", "config/default.yaml")

MODELS = {
    "DecisionTreeClassifier": DecisionTreeClassifier,
    "RandomForestClassifier": RandomForestClassifier,
}


def load_config(config_ref: str) -> dict:
    if config_ref.startswith("s3://"):
        # Get config file from S3
        config_file = Path("config/downloaded-config.yaml")
        try:
            bucket, key = re.match(r"s3://([^/]+)/(.+)", config_ref).groups()
            aws.download_s3(bucket, key, config_file)
        except AttributeError:  # If re.match() does not return groups
            print("Could not parse S3 URI: ", config_ref)
            config_file = Path("config/default.yaml")
        except botocore.exceptions.ClientError as e:  # If there is an error downloading
            print("Unable to download config file from S3: ", config_ref)
            print(e)
    else:
        # Load config from local path
        config_file = Path(config_ref)
    if not config_file.exists():
        raise EnvironmentError(f"Config file at {config_file.absolute()} does not exist")

    with config_file.open() as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def main():
    """Run the pipeline to produce a classifier model for the Iris dataset"""
    # Load the Iris dataset
    iris = load_iris(return_X_y=False)

    config = load_config(CONFIG_REF)
    run_config = config.get("run_config", {})
    print(run_config)
    version = run_config.get("version", "default")

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, **config.get("train_test_split", {})
    )

    # Get config sections; default to empty dict if section is missing
    model_config = config.get("model", {})

    # Create a decision tree classifier
    model = MODELS[model_config.get("type", "DecisionTreeClassifier")]
    clf = model(**model_config.get("params", {}))

    # Train the classifier on the training set
    clf.fit(X_train, y_train)

    # Evaluate the classifier on the test set
    score = clf.score(X_test, y_test)
    print(f"Classifier accuracy: {score:.2f}")

    # Save the data and model to the file system
    joblib.dump(iris, artifacts / "iris.joblib")
    joblib.dump(clf, artifacts / "iris_classifier.joblib")

    run_prefix = ARTIFACTS_PREFIX + version + "/"
    aws.upload_files_to_s3(BUCKET_NAME, run_prefix, artifacts)


if __name__ == "__main__":
    print("Starting iris pipeline...")
    main()



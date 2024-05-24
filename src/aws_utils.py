from pathlib import Path
import pandas as pd
import logging
import boto3

logger = logging.getLogger(__name__)


def download_from_raw(filename: str, config: dict) -> None:
    """***

    Args:
        ***

    Returns:
        ***
    """
    s3_client = boto3.client("s3")

    bucket_name: str = config["bucket_name"]
    object_key: Path = config["raw_prefix"] / filename
    download_path: str = "data" / filename

    try:
        s3_client.download_file(bucket_name, object_key, download_path)
        print(f"File downloaded successfully to {download_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")

def upload_to_refined(filename: str, config: dict) -> str:
    """***

    Args:
        ***

    Returns:
        ***
    """

    s3_client = boto3.client("s3")

    
    bucket_name: str = config["bucket_name"]
    prefix: str = config["refined_prefix"]
    uri: str = ""

    file_path: str = f"data/{filename}"
    key = f"{prefix}/{filename}"
    try:
        s3_client.upload_file(file_path, bucket_name, key)
    except boto3.exceptions.S3UploadFailedError as exc:
        logger.error("The specified bucket does not exist.")
        raise boto3.exceptions.S3UploadFailedError from exc
    uri = f"s3://{bucket_name}/{key}"
    logging.info("Uploaded %s to %s", file_path, uri)

    return uri


import os
from pathlib import Path
import logging
import boto3

logger = logging.getLogger(__name__)

def download_refined_data(bucket_name, s3_key, local_path):
    """Download refined data from S3."""
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_key, str(local_path))
        logger.info("Downloaded refined data from S3.")
    except Exception as e:
        logger.error("Error downloading refined data from S3: %s", e)
        raise

def upload_artifacts(artifacts_path, bucket_name, s3_prefix):
    """Upload artifacts to S3."""
    s3 = boto3.client('s3')
    try:
        for root, dirs, files in os.walk(artifacts_path):
            for file in files:
                file_path = Path(root) / file
                s3_key = f"{s3_prefix}/{file_path.relative_to(artifacts_path)}"
                s3.upload_file(str(file_path), bucket_name, s3_key)
                logger.info(f"Uploaded {file_path} to s3://{bucket_name}/{s3_key}")
    except Exception as e:
        logger.error("Error uploading artifacts to S3: %s", e)
        raise

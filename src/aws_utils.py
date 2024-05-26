import boto3
import os
import logging

logger = logging.getLogger(__name__)

def download_refined_data(bucket_name, s3_key, local_path):
    """
    Download refined data from S3 to a local path.
    """
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        logger.info("Downloaded %s from S3 bucket %s to %s", s3_key, bucket_name, local_path)
    except Exception as e:
        logger.error("Error downloading %s from S3 bucket %s: %s", s3_key, bucket_name, e)

def upload_artifacts(artifacts_dir, bucket_name, s3_prefix):
    """
    Upload all artifacts in the specified directory to an S3 bucket.
    """
    s3 = boto3.client('s3')
    try:
        for root, dirs, files in os.walk(artifacts_dir):
            for file in files:
                local_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_path, artifacts_dir)
                s3_path = os.path.join(s3_prefix, relative_path)
                s3.upload_file(local_path, bucket_name, s3_path)
                logger.info("Uploaded %s to s3://%s/%s", local_path, bucket_name, s3_path)
    except Exception as e:
        logger.error("Error uploading artifacts from %s to S3 bucket %s: %s", artifacts_dir, bucket_name, e)

import boto3
import logging
from botocore.exceptions import NoCredentialsError, ClientError

logger = logging.getLogger(__name__)

def download_refined_data(bucket_name, s3_key, local_path):
    s3 = boto3.client('s3')
    try:
        s3.download_file(bucket_name, s3_key, str(local_path))
        logger.info("Successfully downloaded refined data from S3.")
    except NoCredentialsError:
        logger.error("Credentials not available.")
        raise
    except ClientError as e:
        logger.error("Client error: %s", e)
        raise
    except Exception as e:
        logger.error("Error downloading refined data: %s", e)
        raise

def upload_artifacts(artifacts_path, bucket_name, prefix):
    s3 = boto3.client('s3')
    for root, dirs, files in os.walk(artifacts_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, artifacts_path)
            s3_key = os.path.join(prefix, relative_path)

            try:
                s3.upload_file(local_path, bucket_name, s3_key)
                logger.info("Successfully uploaded %s to S3 as %s.", local_path, s3_key)
            except NoCredentialsError:
                logger.error("Credentials not available.")
                raise
            except ClientError as e:
                logger.error("Client error: %s", e)
                raise
            except Exception as e:
                logger.error("Error uploading %s to S3: %s", local_path, e)
                raise

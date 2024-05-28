import json
import base64
import logging
from PIL import Image
from io import BytesIO
import boto3

logging.basicConfig(level=logging.INFO)


def download_model(bucket_name, s3_key, local_path):
    """Download refined data from S3."""
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket_name, s3_key, str(local_path))
        logging.info("Downloaded refined data from S3.")
    except Exception as e:
        logging.error("Error downloading refined data from S3: %s", e)
        raise


def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def invoke_lambda(client, image_data, lambda_function_name):
    payload = {
        'image_data': image_data
    }

    response = client.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )

    response_payload = json.loads(response['Payload'].read())
    return response_payload


def decode_image(response):
    encoded = json.loads(response["body"])["standardized_image_data"]
    return base64.b64decode(encoded)

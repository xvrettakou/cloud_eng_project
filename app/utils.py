import json
import base64
from io import BytesIO
import logging
from PIL import Image
import boto3

logging.basicConfig(level=logging.INFO)


def download_model(bucket_name: str, s3_key: str, local_path: str) -> None:
    """Download a model file from an S3 bucket to a local path.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    s3_key (str): The key (path) of the model file in the S3 bucket.
    local_path (str): The local path where the model file will be saved.

    Raises:
    Exception: If there is an error during the download process.
    """
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket_name, s3_key, str(local_path))
        logging.info("Downloaded model from S3.")
    except Exception as e:
        logging.error("Error downloading model from S3: %s", e)
        raise


def encode_image(image: Image.Image) -> str:
    """Encode a PIL image as a base64 string.

    Parameters:
    image (Image.Image): The image to be encoded.

    Returns:
    str: The base64 encoded string of the image.
    """
    logging.info("Encoding image")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def invoke_lambda(client: boto3.client, image_data: str, lambda_function_name: str) -> dict:
    """Invoke an AWS Lambda function with the provided image data.

    Parameters:
    client (boto3.client): The boto3 client for Lambda.
    image_data (str): The base64 encoded image data to be sent to the Lambda function.
    lambda_function_name (str): The name of the Lambda function to invoke.

    Returns:
    dict: The response payload from the Lambda function.
    """
    logging.info("Invoking Lambda to process the image")
    payload = {
        "image_data": image_data
    }

    response = client.invoke(
        FunctionName=lambda_function_name,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload)
    )

    response_payload = json.loads(response["Payload"].read())
    return response_payload


def decode_image(response: dict) -> bytes:
    """Decode the base64 encoded image data from the Lambda response.

    Parameters:
    response (dict): The response from the Lambda function, containing the base64 encoded image data.

    Returns:
    bytes: The decoded image data.
    """
    logging.info("Decoding image")
    encoded = json.loads(response["body"])["standardized_image_data"]
    return base64.b64decode(encoded)

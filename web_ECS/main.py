import boto3
import json
import base64
from io import BytesIO
from PIL import Image

# Initialize the Lambda client
lambda_client = boto3.client('lambda')

def read_image(image_path):
    with open(image_path, 'rb') as img_file:
        image_data = base64.b64encode(img_file.read()).decode('utf-8')
    return image_data

def invoke_lambda(image_data, lambda_function_name):
    payload = {
        'image_data': image_data
    }
    response = lambda_client.invoke(
        FunctionName=lambda_function_name,
        InvocationType='RequestResponse',
        Payload=json.dumps(payload)
    )
    response_payload = json.loads(response['Payload'].read())
    return response_payload

def save_image(base64_image, output_path):
    image_data = base64.b64decode(base64_image)
    with open(output_path, 'wb') as img_file:
        img_file.write(image_data)

def main(image_path, lambda_function_name, output_path):
    # Read the image and encode it to base64
    image_data = read_image(image_path)
    
    # Invoke the Lambda function
    print("Invoking Lambda function...")
    response_payload = invoke_lambda(image_data, lambda_function_name)
    
    if response_payload['statusCode'] == 200:
        processed_image_data = json.loads(response_payload['body'])['processed_image_data']
        save_image(processed_image_data, output_path)
        print(f"Processed image saved to: {output_path}")
    else:
        print("Error:", response_payload['body'])

if __name__ == "__main__":
    # Example usage
    image_path = "test.jpg"
    lambda_function_name = "your-lambda-function-name"
    output_path = "processed_image.png"
    
    main(image_path, lambda_function_name, output_path)

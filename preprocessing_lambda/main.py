import csv
import io
import os
import boto3
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

s3 = boto3.client('s3')

# Data augmentation function
def augment_image(image):
    '''Augment a given image with rotation and reflection'''
    augmented_images:list = [image]
    augmented_images.append(ImageOps.mirror(image))
    augmented_images.append(image.rotate(15))
    augmented_images.append(image.rotate(-15))
    return augmented_images

# Lambda handler
def lambda_handler(event, context):
    '''Data agumentation from csv in S3 to new csv in S3'''
    source_bucket = event['Records'][0]['s3']['bucket']['name']
    source_key = event['Records'][0]['s3']['object']['key']
    dest_bucket = 'udn3315-test-0'
    dest_key = f'augmented_{os.path.basename(source_key)}'

    # Download CSV file from S3
    response = s3.get_object(Bucket=source_bucket, Key=source_key)
    csv_content = response['Body'].read().decode('utf-8')

    # Read CSV content
    dataframe = pd.read_csv(io.StringIO(csv_content))

    # Create an output buffer
    output_buffer = io.StringIO()
    writer = csv.writer(output_buffer)
    writer.writerow(['emotion', 'pixels'])

    for index, row in dataframe.iterrows():
        emotion = row['emotion']
        pixels = list(map(int, row[' pixels'].split()))
        image = Image.fromarray(np.array(pixels).reshape(48, 48).astype('uint8'))

        augmented_images = augment_image(image)

        for aug_image in augmented_images:
            aug_pixels = np.array(aug_image).flatten()
            aug_pixels_str = ' '.join(map(str, aug_pixels))
            writer.writerow([emotion, aug_pixels_str])

    # Upload the augmented data to the destination S3 bucket
    s3.put_object(Bucket=dest_bucket, Key=dest_key, Body=output_buffer.getvalue())

    return {
        'statusCode': 200,
        'body': f'Augmented data saved to s3://{dest_bucket}/{dest_key}'
    }

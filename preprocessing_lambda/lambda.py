import os
import csv
import io
import boto3
import numpy as np
from PIL import Image, ImageOps

s3 = boto3.client('s3')

# Data augmentation function
def augment_image(image):
    '''Create additional augmenting images for a given image'''
    augmented_images = [image]
    augmented_images.append(ImageOps.mirror(image))
    augmented_images.append(image.rotate(15))
    augmented_images.append(image.rotate(-15))
    return augmented_images

# Lambda handler
def lambda_handler(event, context):
    '''Read in image csv from S3 and augmented images csv to S3'''
    source_bucket = event['Records'][0]['s3']['bucket']['name']
    source_key = event['Records'][0]['s3']['object']['key']
    dest_bucket = 'udn3315-test-0'
    dest_key = f'augmented_{os.path.basename(source_key)}'

    # Download CSV file from S3
    response = s3.get_object(Bucket=source_bucket, Key=source_key)
    csv_content = response['Body'].read().decode('utf-8')

    # Create an output buffer
    output_buffer = io.StringIO()
    writer = csv.writer(output_buffer)
    writer.writerow(['emotion', 'pixels'])

    # Process CSV content using DictReader to handle headers
    reader = csv.DictReader(io.StringIO(csv_content))
    for row in reader:
        emotion = row['emotion'].strip()
        pixels_str = row['pixels'].strip()

        try:
            pixels = list(map(int, pixels_str.split()))
        except ValueError:
            # If there's an issue with converting pixels, log and skip the row
            print(f"Skipping invalid row: {row}")
            continue

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

import json
import boto3
import pandas as pd
from io import StringIO
from PIL import Image
import numpy as np

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Extract bucket name and object key from the event
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']

    # Download the CSV file from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    csv_content = response['Body'].read().decode('utf-8')

    # Read the CSV content into a pandas DataFrame
    df = pd.read_csv(StringIO(csv_content))

    # Split the data into train, public test, and private test datasets
    train_df = df[df[' Usage'] == 'Training']
    pubtest_df = df[df[' Usage'] == 'PublicTest']
    privtest_df = df[df[' Usage'] == 'PrivateTest']

    # Data augmentation for the training set
    augmented_images = []
    for _, row in train_df.iterrows():
        pixels = list(map(int, row['pixels'].split()))
        image = Image.fromarray(np.array(pixels).reshape(48, 48))
        augmented_images.extend(augment_image(image))

    augmented_train_df = pd.DataFrame(augmented_images, columns=train_df.columns)

    # Combine original and augmented data
    final_train_df = pd.concat([train_df, augmented_train_df], ignore_index=True)

    # Save dataframes back to CSV and upload to S3
    upload_csv_to_s3(final_train_df, bucket, 'train_augmented.csv')
    upload_csv_to_s3(pubtest_df, bucket, 'public_test.csv')
    upload_csv_to_s3(privtest_df, bucket, 'private_test.csv')

    return {
        'statusCode': 200,
        'body': json.dumps('Data processed and uploaded successfully')
    }

def augment_image(image):
    augmented_images = []
    transformations = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    for transform in transformations:
        augmented_image = image.transpose(transform)
        augmented_images.append({
            'emotion': image['emotion'],
            'Usage': 'Training',
            'pixels': ' '.join(map(str, augmented_image.flatten().tolist()))
        })

    return augmented_images

def upload_csv_to_s3(df, bucket, key):
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
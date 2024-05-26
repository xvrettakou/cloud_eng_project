import json


def lambda_handler(event, context):
    print(event)
    uris = []
    for record in event["Records"]:
        message_body = json.loads(record["body"])
        bucket_name = message_body["detail"]["bucket"]["name"]
        object_key = message_body["detail"]["object"]["key"]
        uris.append(f"s3://{bucket_name}/{object_key}")
        print(f"Object {object_key} created in bucket {bucket_name}.")


        # do something with the object, such as copy it to another S3 bucket or process it

        

        if object_key.contains('train'):
            download_path = '/tmp/{}'.format(os.path.basename(object_key))
            try:
                s3_client.download_file(bucket_name, object_key, download_path)
                print(f"File downloaded successfully to {download_path}")
            except Exception as e:
                print(f"Error downloading file: {e}")

            
        elif object_key.contains('test'):
            dfas
        else:
            return{
        "statusCode": 0,
        "body": json.dumps("File was not marked as either test or train."),
    }


    return {
        "statusCode": 200,
        "body": json.dumps("Successfully processed S3 notifications from EventBridge."),
    }
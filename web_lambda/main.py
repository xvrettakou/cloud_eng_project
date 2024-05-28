import json
import base64
import numpy as np
from PIL import Image
from io import BytesIO

def resize_image(image, target_size=(48, 48)):
    original_width, original_height = image.size
    target_width, target_height = target_size
    scaling_factor = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    new_image = Image.new("L", target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    return new_image

def lambda_handler(event, context):
    try:
        # Extract base64 encoded image data from the event
        image_data = event['image_data']
        # Decode the base64 encoded image data to bytes
        image_data = base64.b64decode(image_data)
        
        # Load the image from bytes
        image = Image.open(BytesIO(image_data)).convert('L')
        
        # Resize the image to (48, 48) without distortion
        image = resize_image(image, target_size=(48, 48))
        
        # Convert the image to a numpy array and reshape to (48, 48, 1)
        image_array = np.array(image).reshape((48, 48, 1))
        
        # Convert the numpy array back to base64
        buffered = BytesIO()
        image = Image.fromarray(image_array.squeeze(), 'L')
        image.save(buffered, format="PNG")
        standardized_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Image processed successfully',
                'standardized_image_data': standardized_image_data
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error processing image',
                'error': str(e)
            })
        }
# pylint: disable=no-member
import json
import base64
from io import BytesIO
from typing import Tuple, Dict, Any
import numpy as np
from PIL import Image

def resize_image(image: Image.Image, target_size: Tuple[int, int] = (48, 48)) -> Image.Image:
    """Resize the image to the target size without distortion.
    
    Args:
        image (PIL.Image.Image): The original image.
        target_size (tuple): The target size as (width, height).
    
    Returns:
        PIL.Image.Image: The resized image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_size
    scaling_factor = min(target_width / original_width, target_height / original_height)
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)  # pylint: disable=no-member
    new_image = Image.new('L', target_size)
    new_image.paste(resized_image, ((target_width - new_width) // 2, (target_height - new_height) // 2))
    return new_image

def lambda_handler(event: Dict[str, Any], _context: Any) -> Dict[str, Any]:
    """AWS Lambda handler function to process the image.
    
    Args:
        event (dict): The event payload containing base64 image data.
        _context (LambdaContext): The context in which the function is called.
    
    Returns:
        dict: The response with status code and processed image data.
    """
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
        image.save(buffered, format='PNG')
        standardized_image_data = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Image processed successfully',
                'standardized_image_data': standardized_image_data
            })
        }
    except (ValueError, KeyError, IOError) as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error processing image',
                'error': str(e)
            })
        }

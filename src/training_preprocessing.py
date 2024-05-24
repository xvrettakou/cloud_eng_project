import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import cv2


logger = logging.getLogger(__name__)

def get_df(file_path: str) -> pd.DataFrame:
    """***

    Args:
        ***

    Returns:
        ***
    """
    
    try:
        df: pd.DataFrame = pd.read_csv(file_path)
    except Exception as e:
        logger.error("Specified path is not valid: %s", file_path)
    return df
    
def preprocess_pixels(pixels: np.array) -> np.array:
    """***

    Args:
        ***

    Returns:
        ***
    """
    return np.array(pixels.split(), dtype='float32').reshape(48, 48, 1)

# Simple image augmentation function
def augment_image(image: np.array) -> list:
    """***

    Args:
        ***

    Returns:
        ***
    """
    augmented_images: list = []

    # Original image
    augmented_images.append(image)

    # Rotate
    for angle in [10, -10]:
        M: np.matrix = cv2.getRotationMatrix2D((24, 24), angle, 1)
        rotated: np.array = cv2.warpAffine(image, M, (48, 48))
        augmented_images.append(rotated.reshape(48, 48, 1))

    # Flip horizontally
    flipped: np.array = cv2.flip(image, 1)
    augmented_images.append(flipped.reshape(48, 48, 1))

    # Shift
    for x_shift, y_shift in [(5, 0), (-5, 0), (0, 5), (0, -5)]:
        M: np.matrix = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        shifted: np.array = cv2.warpAffine(image, M, (48, 48))
        augmented_images.append(shifted.reshape(48, 48, 1))

    return augmented_images
    

def save_dataframe(dataset: pd.DataFrame , save_path: Path) -> None:
    """Save the dataset locally.

    Args:
        dataset: Dataframe of dataset
        save_path: Local directory to save dataset to

    """
    try:
        dataset.to_csv(save_path, index=False)
    except FileNotFoundError as exc:
        logger.error("Specified path is not valid: %s", save_path)
        raise FileNotFoundError(f"Specified save path '{save_path}' does not exist.") from exc
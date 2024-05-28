import logging
import os
from io import BytesIO
import boto3
import joblib
import streamlit as st
from PIL import Image
import numpy as np
from sklearn.base import BaseEstimator
import utils


logging.basicConfig(level=logging.INFO)
lambda_client = boto3.client("lambda", region_name="us-east-2")
lambda_function_name = os.getenv(
    "LAMBDA_FUNCTION_NAME",
    "Inference-ImageProcess")
bucket_name = os.getenv("BUCKET_NAME", "cloud-project-artifact")


# Define the emotion labels
emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"]


# Load the trained model
@st.cache_resource
def load_model(version):
    """Load the specified model version from a pickle file.

    Args:
    version (str): The version of the model to load.

    Returns:
    model: The loaded model.
    """
    clf = None
    try:
        if version == "Random Forest":
            s3_key = "trained_model.pkl"
            utils.download_model(
                bucket_name,
                "artifacts/trained_model.pkl/" + s3_key,
                s3_key)

            with open("trained_model.pkl", "rb") as file:
                clf = joblib.load(file)

    except FileNotFoundError as e:
        logging.error("Model file not found: %s", e)

    return clf


st.set_page_config(page_title="Facial Expression Recognition")
st.title("Facial Expression Recognition", anchor=False)
st.caption("This app uses a trained model to classify facial expressions\
    into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.")
st.subheader("Select an image source to proceed:", anchor=False)

MODEL_VERSION = "Random Forest"

model = load_model(MODEL_VERSION)
image = None


# Option selection
option = st.radio(
    "Choose your image source:",
    ("Upload an Image",
     "Capture from Webcam"))

if option == "Upload an Image":
    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader(
        "Choose an image...", type=[
            "jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.")

elif option == "Capture from Webcam":
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)


if image is not None:
    encoded = utils.encode_image(image)
    response = utils.invoke_lambda(
        lambda_client, encoded, lambda_function_name)

    if response["statusCode"] == 200:
        decoded = utils.decode_image(response)
        image = Image.open(BytesIO(decoded))


if image is not None and model is not None:
    logging.info("Making prediction")

    if isinstance(model, BaseEstimator):
        img_array = np.array(image).reshape(1, -1)
        prediction = model.predict(img_array)[0]
        predicted_emotion = emotion_labels[prediction]
    else:
        img_array = np.array([image])
        prediction = model.predict(img_array)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

    st.subheader(f"The predicted emotion is: {predicted_emotion}")

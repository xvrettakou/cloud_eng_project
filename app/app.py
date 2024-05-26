import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
st.set_page_config(page_title="Facial Expression Recognition")

# Load the pre-trained TensorFlow model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("baseline_epochs=100.h5")
    return model


def preprocess_image(img_tensor):
    img_tensor = tf.image.resize_with_crop_or_pad(img_tensor, 48, 48)

    # Convert image to grayscale
    image = tf.image.rgb_to_grayscale(img_tensor)

    image = image.numpy()

    return image


st.title('Facial Expression Recognition', anchor=False)
st.caption("This app uses a pre-trained model to classify facial expressions into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.")
st.subheader('Select an image source to proceed:', anchor=False)

model = load_model()
image = None

# Option selection
option = st.radio("Choose your image source:", ('Upload an Image', 'Capture from Webcam'))

if option == 'Upload an Image':
    # File uploader allows user to add their own image
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.')

elif option == 'Capture from Webcam':
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        img_tensor = tf.io.decode_image(bytes_data, channels=3)

        # Check the type of img_tensor:
        # Should output: <class 'tensorflow.python.framework.ops.EagerTensor'>
        # st.write(type(img_tensor))

        # Check the shape of img_tensor:
        # Should output shape: (height, width, channels)
        # st.write(img_tensor.shape)

        # Preprocess the image
        # (Resize, normalize, etc.)
        gray_image = preprocess_image(img_tensor)

        print(type(gray_image))
        print(gray_image)

        st.image(
            gray_image,
            caption='Preprocessed Image',
            use_column_width=True
        )


        # Make a prediction
        #prediction = model.predict(gray_image)

        # Get the emotion label with the highest probability
        #predicted_emotion = emotion_labels[np.argmax(prediction)]

        # Display the result
        #st.write(f"The predicted emotion is: {predicted_emotion}")


if image is not None:
    img_array = np.array([image])
    print(img_array.shape)
    prediction = model.predict(img_array)
    predicted_emotion = emotion_labels[np.argmax(prediction)]
    st.subheader(f"The predicted emotion is: {predicted_emotion}")

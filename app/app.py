import streamlit as st
import tensorflow as tf


# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
st.set_page_config(page_title="Facial Expression Recognition")


def main():
    st.title('Facial Expression Recognition', anchor=False)
    st.caption("This app uses a pre-trained model to classify facial expressions into one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.")
    st.subheader('Select an image source to proceed:', anchor=False)

    # Option selection
    option = st.radio("Choose your image source:", ('Upload an Image', 'Capture from Webcam'))

    if option == 'Upload an Image':
        # File uploader allows user to add their own image
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image.', use_column_width=True)

    elif option == 'Capture from Webcam':
        img_file_buffer = st.camera_input("Take a picture")

        if img_file_buffer is not None:
            bytes_data = img_file_buffer.getvalue()
            img_tensor = tf.io.decode_image(bytes_data, channels=3)

            # Check the type of img_tensor:
            # Should output: <class 'tensorflow.python.framework.ops.EagerTensor'>
            st.write(type(img_tensor))

            # Check the shape of img_tensor:
            # Should output shape: (height, width, channels)
            st.write(img_tensor.shape)

    
    # # Preprocess the image
    # # (Resize, normalize, etc.)
    # preprocessed_image = preprocess_image(opencv_image)

    # # Make a prediction
    # prediction = model.predict(preprocessed_image)

    # # Get the emotion label with the highest probability
    # predicted_emotion = emotion_labels[np.argmax(prediction)]

    # # Display the result
    # st.write(f"The predicted emotion is: {predicted_emotion}")



if __name__ == "__main__":
    main()
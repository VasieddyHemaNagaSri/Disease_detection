import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive Model File
file_id = "1_9Q28QeJXnyRCH18IVxpX5KhuDtZC40c"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Function to Download Model
def download_model():
    if not os.path.exists(model_path):
        st.warning("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
        
        if os.path.exists(model_path):
            st.success("Model downloaded successfully!")
        else:
            st.error("Model download failed. Check the file ID and permissions.")

download_model()  # Ensure the model is downloaded

# Load Model with Error Handling
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(model_path):
            st.error("Model file not found. Ensure the file is downloaded correctly.")
            return None

        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model

    except ValueError as ve:
        st.error(f"ValueError while loading model: {ve}")
    except Exception as e:
        st.error(f"Unexpected error while loading model: {e}")

    return None

model = load_model()

# Function for Model Prediction
def model_prediction(image_file):
    try:
        if model is None:
            st.error("Model is not loaded. Cannot make predictions.")
            return None

        image = Image.open(image_file).convert("RGB")
        image = image.resize((128, 128))
        input_arr = np.array(image) / 255.0  # Normalize image
        input_arr = np.expand_dims(input_arr, axis=0)

        predictions = model.predict(input_arr)
        return np.argmax(predictions)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Sidebar Navigation
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display Banner Image
banner_img = Image.open("Diseses.png")
st.image(banner_img, use_column_width=True)

# Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System")

    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict") and test_image:
        st.snow()  # Fun effect
        st.write("Our Prediction:")
        
        result_index = model_prediction(test_image)
        
        if result_index is not None:
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f"Model is Predicting: {class_name[result_index]}")

import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Corrected Google Drive direct download link
file_id = "1r6O6VvfVIjzUqJ2QBOVjFv8B-O4DbbVA"
url = f"https://drive.google.com/uc?id={file_id}"  

# Path to store model
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

# Call function to ensure model is downloaded
download_model()

# Function to Load Model
@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            st.success("Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None
    else:
        st.error("Model file not found. Ensure it's correctly downloaded.")
        return None

# Load Model
model = load_model()

# Function for Model Prediction
def model_prediction(model, test_image):
    try:
        image = Image.open(test_image).convert("RGB")
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
img = Image.open("Diseses.png")
st.image(img)

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
        if model:
            result_index = model_prediction(model, test_image)
            if result_index is not None:
                class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
                st.success(f"Model is Predicting: {class_name[result_index]}")
                st.snow()
        else:
            st.error("Model not loaded. Please check the model file.")

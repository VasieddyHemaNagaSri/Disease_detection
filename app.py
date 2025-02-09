import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive File ID & Direct Download URL
file_id = "1r6O6VvfVIjzUqJ2QBOVjFv8B-O4DbbVA"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Ensure Model is Downloaded
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Function to Load Model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function for Model Prediction
def model_prediction(model, test_image):
    try:
        image = Image.open(test_image).convert("RGB")
        image = image.resize((128, 128))
        input_arr = np.array(image) / 255.0  # Normalize the image
        input_arr = np.expand_dims(input_arr, axis=0)
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Load Model Once
model = load_model()

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

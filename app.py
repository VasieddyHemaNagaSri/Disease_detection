import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# New Google Drive file ID
file_id = "1r6O6VvfVIjzUqJ2QBOVjFv8B-O4DbbVA"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Check and download model
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive... Please wait.")
    gdown.download(url, model_path, quiet=False)

# Load model once (instead of reloading in every function)
try:
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()  # Stop execution if model loading fails

# Function to make predictions
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")  # Convert to RGB to avoid issues
    image = image.resize((128, 128))  # Resize for model input
    input_arr = np.array(image) / 255.0  # Normalize pixel values
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert to batch format

    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return class index

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display image
img = Image.open("Diseases.png")
st.image(img, use_container_width=True)

# Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>🌱 Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("📸 Upload an Image for Disease Detection")
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)

        if st.button("Predict"):
            st.snow()
            st.write("🧐 Analyzing the Image...")
            result_index = model_prediction(test_image)

            # Define class names
            class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

            # Show result
            st.success(f"✅ Model Prediction: {class_names[result_index]}")

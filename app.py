import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive File ID
file_id = "1r6O6VvfVIjzUqJ2QBOVjFv8B-O4DbbVA"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Download model if it doesn't exist
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive... Please wait.")
    gdown.download(url, model_path, quiet=False)

# Load Model
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path)
else:
    st.error("Model file not found. Please check the Google Drive link and ensure the file is accessible.")

# Model Prediction Function
def model_prediction(test_image):
    image = Image.open(test_image)
    image = image.resize((128, 128))  # Resize to match model input size
    input_arr = np.array(image) / 255.0  # Normalize the image
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max probability

# Sidebar
st.sidebar.title("🌱 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["🏠 HOME", "🦠 DISEASE RECOGNITION"])

# Display an image
img = Image.open("Diseses.png")
st.image(img, use_container_width=True)

# Home Page
if app_mode == "🏠 HOME":
    st.markdown("<h1 style='text-align: center;'>🌿 Plant Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "🦠 DISEASE RECOGNITION":
    st.header("🔍 Upload a Plant Image for Disease Detection")
    
    test_image = st.file_uploader("📂 Choose an Image (JPG, PNG)", type=["jpg", "png", "jpeg"])
    
    if test_image:
        st.image(test_image, caption="Uploaded Image", use_container_width=True)
    
        if st.button("🚀 Predict"):
            st.snow()
            result_index = model_prediction(test_image)

            # Class Labels
            class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            prediction_text = class_names[result_index]

            st.success(f"✅ Model Prediction: {prediction_text}")

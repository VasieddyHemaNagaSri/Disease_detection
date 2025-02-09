import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive File ID and Model Path
file_id = "1r6O6VvfVIjzUqJ2QBOVjFv8B-O4DbbVA"
model_path = "trained_plant_disease_model.keras"

# Download model if not exists
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(id=file_id, output=model_path, quiet=False)

# Load Model Once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(model_path)

model = load_model()

# Model Prediction Function
def model_prediction(test_image):
    image = Image.open(test_image).convert("RGB")
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0  # Normalize image
    input_arr = np.expand_dims(input_arr, axis=0)
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Sidebar
st.sidebar.title("Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display Header Image
img = Image.open("Diseses.png")
st.image(img, use_column_width=True)

# Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", 
                unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System")

    # File Uploader
    test_image = st.file_uploader("Choose an Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            st.snow()
            result_index = model_prediction(test_image)
            class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']
            st.success(f"Model is predicting it's a **{class_names[result_index]}**")

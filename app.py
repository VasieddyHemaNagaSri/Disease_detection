import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image

# Model file path
model_path = "trained_plant_disease_model.keras"

# Function to check and download model
def check_and_download_model():
    url = "https://drive.google.com/uc?export=download&id=1r6O6VvfVIjzUqJ2QBOVjFv8B-O4DbbVA"
    
    if not os.path.exists(model_path):
        st.warning("Downloading model from Google Drive... Please wait.")
        os.system(f"wget -O {model_path} '{url}'")

        # Verify if the model was downloaded
        if os.path.exists(model_path):
            st.success("✅ Model downloaded successfully!")
        else:
            st.error("❌ Model download failed! Please upload it manually.")
            return False
    return True

# Check if the model exists or needs to be downloaded
if check_and_download_model():
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("🚀 Model Loaded Successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

# Function to make predictions
def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max probability

# Sidebar
st.sidebar.title("🌿 Plant Disease Detection System")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display image on the main page
img = Image.open("Diseses.png")
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

            # Define class names (Update if your dataset has different classes)
            class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

            # Show the result
            st.success(f"✅ Model Prediction: {class_names[result_index]}")

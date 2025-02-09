import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive File ID
file_id = "1wnlQz-U7_6P-y8SZuZQxTYtR5Hfm1OQd"
url = f"https://drive.google.com/uc?id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Function to download model
def download_model():
    if not os.path.exists(model_path):
        st.warning("Downloading model from Google Drive...")
        gdown.download(url, model_path, quiet=False)
    
    # Check if model is successfully downloaded
    if os.path.exists(model_path):
        st.success("✅ Model downloaded successfully!")
        st.write("📂 Files in current directory:", os.listdir())  # Debugging
        return True
    else:
        st.error("⚠️ Model download failed! Please check the file link.")
        return False

# Download & Verify Model
if download_model():
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        model = None
else:
    model = None

# Prediction Function
def model_prediction(test_image):
    try:
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"⚠️ Unable to make prediction: {e}")
        return None

# Sidebar & UI
st.sidebar.title("🌱 Plant Disease Detection")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Show Image
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection</h1>", unsafe_allow_html=True)
elif app_mode == "DISEASE RECOGNITION":
    st.header("🌿 Plant Disease Detection")
    test_image = st.file_uploader("📸 Upload an Image:")

    if test_image and st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    
    if model and test_image and st.button("🔍 Predict"):
        st.snow()
        result_index = model_prediction(test_image)

        class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___Healthy']
        if result_index is not None:
            st.success(f"🌱 Model predicts: **{class_name[result_index]}**")

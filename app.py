import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os

# New Google Drive File ID
file_id = "1wnlQz-U7_6P-y8SZuZQxTYtR5Hfm1OQd"
url = f"https://drive.google.com/uc?id={file_id}"

# Model file path
model_path = "trained_plant_disease_model.keras"

# Download model if not exists
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive... Please wait.")
    try:
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")

        # Debugging: Show files in the directory
        st.write("Files in the current directory:", os.listdir())

    except Exception as e:
        st.error(f"⚠️ Failed to download model: {e}")
        st.stop()

# Function to load and make predictions
def model_prediction(test_image):
    try:
        model = tf.keras.models.load_model(model_path)
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions)  # Return index of max element
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display image
from PIL import Image
img = Image.open("Diseases.png")
st.image(img)

# Home Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)

        if result_index is not None:
            class_name = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
            st.success(f"Model is predicting it's a {class_name[result_index]}")
        else:
            st.error("⚠️ Unable to make prediction. Check model loading.")

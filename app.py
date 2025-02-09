import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
from PIL import Image

# Google Drive File ID (Make sure it's publicly accessible)
file_id = "1r6O6VvfVIjzUqJ2QBOVjFv8B-O4DbbVA"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
model_path = "trained_plant_disease_model.keras"

# Ensure the model is downloaded correctly
if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive... Please wait.")
    try:
        gdown.download(url, model_path, quiet=False)
        st.success("✅ Model downloaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to download model: {e}")
        st.stop()  # Stop execution if download fails

# Try loading the model
try:
    model = tf.keras.models.load_model(model_path)
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()  # Stop execution if model loading fails

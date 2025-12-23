import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Plant Disease Recognition", layout="centered")

# -----------------------------
# Paths (FIXED)
# -----------------------------
MODEL_PATH = "plant_disease_mobilenetV2_model.keras"
CLASS_PATH = "class_names.json"

# -----------------------------
# Load model & classes
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_PATH):
    model = load_model()
    class_names = load_classes()
    st.success("Model & class labels loaded ✅")
else:
    st.error("Model or class_names.json not found ❌")
    st.stop()

# -----------------------------
# Prediction function
# -----------------------------
def model_prediction(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))

    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    return class_names[index], confidence

# -----------------------------
# UI
# -----------------------------
st.title("🌿 Plant Disease Recognition")

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    label, conf = model_prediction(uploaded_file)

    st.subheader("Prediction")
    st.write(f"**Disease:** {label}")
    st.write(f"**Confidence:** {conf:.2%}")

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Plant Disease Recognition", layout="centered")

MODEL_PATH = "plant_disease_mobilenetV2_model.keras"
CLASS_PATH = "class_names.json"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASS_PATH):
    st.error("Model or class_names.json not found")
    st.stop()

model = load_model()
class_names = load_classes()

def predict_image(image_file):
    image = Image.open(image_file).convert("RGB")
    image = image.resize((224, 224))

    img = np.array(image)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    preds = model.predict(img)
    idx = np.argmax(preds)
    return class_names[idx], float(preds[0][idx])

st.title("🌿 Plant Disease Recognition")

uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    label, conf = predict_image(uploaded_file)
    st.success(f"Disease: {label}")
    st.write(f"Confidence: {conf:.2%}")

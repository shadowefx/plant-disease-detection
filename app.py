import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and class names
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_names.json') as f:
    class_names = json.load(f)

st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the disease")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Leaf", width=300)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    st.success(f"🔍 Detected: **{predicted_class}**")
    st.info(f"Confidence: {confidence}%")
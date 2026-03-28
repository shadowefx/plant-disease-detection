import streamlit as st
import numpy as np
from PIL import Image
import json
import tf_keras as keras

# Load model and class names
model = keras.models.load_model('plant_disease_model.h5')

with open('class_names.json') as f:
    class_names = json.load(f)

# App title
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the disease")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((224, 224))
    st.image(image, caption="Uploaded Leaf", width=300)

    # Prepare image for prediction
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    # Show result
    st.success(f"🔍 Detected: **{predicted_class}**")
    st.info(f"Confidence: {confidence}%")

    # Show all predictions
    st.write("### All Predictions:")
    for i, (name, prob) in enumerate(zip(class_names, prediction[0])):
        st.progress(float(prob), text=f"{name}: {round(prob*100, 2)}%")
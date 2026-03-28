import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Disease information dictionary
disease_info = {
    "Pepper__bell__Bacterial_spot": {
        "description": "A bacterial disease causing dark, water-soaked spots on leaves and fruits.",
        "treatment": "Apply copper-based bactericides. Remove infected plants. Avoid overhead watering."
    },
    "Pepper__bell__healthy": {
        "description": "The plant is healthy! No disease detected.",
        "treatment": "Keep up good watering and fertilization practices."
    },
    "Potato__Early_blight": {
        "description": "A fungal disease causing brown spots with yellow rings on older leaves.",
        "treatment": "Apply fungicides containing chlorothalonil. Remove infected leaves. Rotate crops."
    },
    "Potato__Late_blight": {
        "description": "A serious fungal disease causing dark lesions on leaves and stems.",
        "treatment": "Apply fungicides immediately. Remove infected plants. Avoid wet conditions."
    },
    "Potato__healthy": {
        "description": "The plant is healthy! No disease detected.",
        "treatment": "Maintain proper watering and soil nutrition."
    },
    "Tomato__Bacterial_spot": {
        "description": "Bacterial infection causing small, dark spots on leaves and fruits.",
        "treatment": "Use copper-based sprays. Avoid working with wet plants. Remove infected parts."
    },
    "Tomato__Early_blight": {
        "description": "Fungal disease causing target-like brown spots on lower leaves first.",
        "treatment": "Apply fungicides. Remove lower infected leaves. Water at the base of the plant."
    },
    "Tomato__Late_blight": {
        "description": "Destructive fungal disease causing large, dark water-soaked lesions.",
        "treatment": "Apply fungicides immediately. Remove infected plants to prevent spread."
    },
    "Tomato__Leaf_Mold": {
        "description": "Fungal disease causing yellow patches on upper leaf surface and mold below.",
        "treatment": "Improve air circulation. Apply fungicides. Reduce humidity in greenhouse."
    },
    "Tomato__Septoria_leaf_spot": {
        "description": "Fungal disease causing small circular spots with dark borders on leaves.",
        "treatment": "Apply fungicides. Remove infected leaves. Avoid overhead irrigation."
    },
    "Tomato__Spider_mites Two-spotted_spider_mite": {
        "description": "Tiny mites causing yellowing and speckling on leaves.",
        "treatment": "Apply miticides or neem oil. Increase humidity. Remove heavily infested leaves."
    },
    "Tomato__Target_Spot": {
        "description": "Fungal disease causing circular lesions with concentric rings.",
        "treatment": "Apply fungicides. Ensure good air circulation. Avoid leaf wetness."
    },
    "Tomato__Tomato_Yellow_Leaf_Curl_Virus": {
        "description": "Viral disease causing yellowing and curling of leaves spread by whiteflies.",
        "treatment": "Control whitefly population. Remove infected plants. Use virus-resistant varieties."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral disease causing mosaic-like yellowing and distortion of leaves.",
        "treatment": "Remove infected plants. Disinfect tools. Control aphid population."
    },
    "Tomato__healthy": {
        "description": "The plant is healthy! No disease detected.",
        "treatment": "Maintain proper watering, fertilization, and sunlight."
    }
}
# Load model and class names
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_names.json') as f:
    class_names = json.load(f)
# App UI
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to detect the disease and get treatment advice.")
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((224, 224))
    st.image(image, caption="Uploaded Leaf", width=300)
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)
    st.success(f"🔍 Detected: **{predicted_class}**")
    st.info(f"Confidence: **{confidence}%**")

    # Disease info
    if predicted_class in disease_info:
        info = disease_info[predicted_class]
        st.write("---")
        st.subheader("📋 Disease Information")
        st.write(f"**About:** {info['description']}")
        st.warning(f"💊 **Treatment:** {info['treatment']}")

    # All predictions
    st.write("---")
    st.subheader("📊 All Predictions")
    for name, prob in zip(class_names, prediction[0]):
        st.progress(float(prob), text=f"{name}: {round(float(prob)*100, 2)}%")
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load model and class names
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_names.json') as f:
    class_names = json.load(f)

# Disease info
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "description": "A bacterial disease causing dark, water-soaked spots on leaves and fruits.",
        "treatment": "Apply copper-based bactericides. Remove infected plants. Avoid overhead watering."
    },
    "Pepper__bell___healthy": {
        "description": "The plant is healthy! No disease detected.",
        "treatment": "Keep up good watering and fertilization practices."
    },
    "Potato___Early_blight": {
        "description": "A fungal disease causing brown spots with yellow rings on older leaves.",
        "treatment": "Apply fungicides containing chlorothalonil. Remove infected leaves. Rotate crops."
    },
    "Potato___Late_blight": {
        "description": "A serious fungal disease causing dark lesions on leaves and stems.",
        "treatment": "Apply fungicides immediately. Remove infected plants. Avoid wet conditions."
    },
    "Potato___healthy": {
        "description": "The plant is healthy! No disease detected.",
        "treatment": "Maintain proper watering and soil nutrition."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial infection causing small, dark spots on leaves and fruits.",
        "treatment": "Use copper-based sprays. Avoid working with wet plants. Remove infected parts."
    },
    "Tomato_Early_blight": {
        "description": "Fungal disease causing target-like brown spots on lower leaves first.",
        "treatment": "Apply fungicides. Remove lower infected leaves. Water at the base of the plant."
    },
    "Tomato_Late_blight": {
        "description": "Destructive fungal disease causing large, dark water-soaked lesions.",
        "treatment": "Apply fungicides immediately. Remove infected plants to prevent spread."
    },
    "Tomato_Leaf_Mold": {
        "description": "Fungal disease causing yellow patches on upper leaf surface and mold below.",
        "treatment": "Improve air circulation. Apply fungicides. Reduce humidity in greenhouse."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Fungal disease causing small circular spots with dark borders on leaves.",
        "treatment": "Apply fungicides. Remove infected leaves. Avoid overhead irrigation."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Tiny mites causing yellowing and speckling on leaves.",
        "treatment": "Apply miticides or neem oil. Increase humidity. Remove heavily infested leaves."
    },
    "Tomato__Target_Spot": {
        "description": "Fungal disease causing circular lesions with concentric rings.",
        "treatment": "Apply fungicides. Ensure good air circulation. Avoid leaf wetness."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease causing yellowing and curling of leaves spread by whiteflies.",
        "treatment": "Control whitefly population. Remove infected plants. Use virus-resistant varieties."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral disease causing mosaic-like yellowing and distortion of leaves.",
        "treatment": "Remove infected plants. Disinfect tools. Control aphid population."
    },
    "Tomato_healthy": {
        "description": "The plant is healthy! No disease detected.",
        "treatment": "Maintain proper watering, fertilization, and sunlight."
    }
}

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

    # Clean display name (remove underscores)
    display_name = predicted_class.replace("_", " ").replace("  ", " ")

    st.success(f"🔍 Detected: **{display_name}**")
    st.info(f"Confidence: **{confidence}%**")

    # Disease info
    if predicted_class in disease_info:
        info = disease_info[predicted_class]
        st.write("---")
        st.subheader("📋 Disease Information")
        st.write(f"**About:** {info['description']}")
        st.warning(f"💊 **Treatment:** {info['treatment']}")
    else:
        st.write("---")
        st.subheader("📋 Disease Information")
        st.write(f"**Detected:** {display_name}")
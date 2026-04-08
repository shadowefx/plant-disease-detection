import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
from datetime import datetime
from fpdf import FPDF
import io
import base64
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f5f9f5; }
    .sidebar .sidebar-content { background-color: #2d6a4f; }
    .title {
        text-align: center;
        color: #2d6a4f;
        font-size: 2.5em;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #52b788;
        font-size: 1.1em;
        margin-bottom: 20px;
    }
    .result-box {
        background-color: #d8f3dc;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #2d6a4f;
        margin: 10px 0;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #457b9d;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #f4a261;
        margin: 10px 0;
    }
    .home-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        text-align: center;
        margin: 10px 0;
    }
    .history-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        margin: 8px 0;
        border-left: 5px solid #52b788;
    }
    .stButton>button {
        background-color: #2d6a4f;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 20px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #52b788;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('plant_disease_model.h5')
    with open('class_names.json') as f:
        class_names = json.load(f)
    return model, class_names

model, class_names = load_model()

# Disease database
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "description": "A bacterial disease causing dark, water-soaked spots on leaves and fruits.",
        "severity": "Moderate",
        "treatment": "Apply copper-based bactericides. Remove infected plants. Avoid overhead watering.",
        "prevention": "Use disease-free seeds. Rotate crops every season. Avoid working with wet plants."
    },
    "Pepper__bell___healthy": {
        "description": "The plant is perfectly healthy! No disease detected.",
        "severity": "None",
        "treatment": "No treatment needed. Keep up the good work!",
        "prevention": "Continue regular watering, fertilization, and monitor for early signs of disease."
    },
    "Potato___Early_blight": {
        "description": "A fungal disease causing brown spots with yellow rings on older leaves.",
        "severity": "Mild",
        "treatment": "Apply fungicides containing chlorothalonil. Remove infected leaves. Rotate crops.",
        "prevention": "Plant certified disease-free tubers. Maintain proper plant spacing for air circulation."
    },
    "Potato___Late_blight": {
        "description": "A serious fungal disease causing dark lesions on leaves and stems.",
        "severity": "Severe",
        "treatment": "Apply fungicides immediately. Remove and destroy infected plants.",
        "prevention": "Use resistant varieties. Apply preventive fungicides during wet weather."
    },
    "Potato___healthy": {
        "description": "The plant is perfectly healthy! No disease detected.",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper soil drainage and avoid over-watering."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial infection causing small, dark spots on leaves and fruits.",
        "severity": "Moderate",
        "treatment": "Use copper-based sprays. Avoid working with wet plants. Remove infected parts.",
        "prevention": "Use pathogen-free seeds. Avoid overhead irrigation. Rotate crops annually."
    },
    "Tomato_Early_blight": {
        "description": "Fungal disease causing target-like brown spots starting on lower leaves.",
        "severity": "Mild",
        "treatment": "Apply fungicides. Remove lower infected leaves. Water at base of plant.",
        "prevention": "Mulch around plants. Stake tomatoes to improve airflow."
    },
    "Tomato_Late_blight": {
        "description": "Destructive fungal disease causing large dark water-soaked lesions.",
        "severity": "Severe",
        "treatment": "Apply fungicides immediately. Remove infected plants to prevent spread.",
        "prevention": "Plant resistant varieties. Apply fungicides before rainy season."
    },
    "Tomato_Leaf_Mold": {
        "description": "Fungal disease causing yellow patches on upper leaf surface with mold below.",
        "severity": "Moderate",
        "treatment": "Improve air circulation. Apply fungicides. Reduce humidity.",
        "prevention": "Maintain low humidity. Space plants properly."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Fungal disease causing small circular spots with dark borders on leaves.",
        "severity": "Moderate",
        "treatment": "Apply fungicides. Remove infected leaves. Avoid overhead irrigation.",
        "prevention": "Rotate crops. Remove plant debris after harvest."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Tiny mites causing yellowing and speckling on leaves.",
        "severity": "Mild",
        "treatment": "Apply miticides or neem oil. Increase humidity.",
        "prevention": "Keep plants well-watered. Introduce natural predators."
    },
    "Tomato__Target_Spot": {
        "description": "Fungal disease causing circular lesions with concentric rings.",
        "severity": "Moderate",
        "treatment": "Apply fungicides. Ensure good air circulation.",
        "prevention": "Prune lower leaves. Use drip irrigation."
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "description": "Viral disease causing yellowing and curling of leaves spread by whiteflies.",
        "severity": "Severe",
        "treatment": "Control whitefly population. Remove infected plants.",
        "prevention": "Use reflective mulches. Install insect nets."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Viral disease causing mosaic-like yellowing and distortion of leaves.",
        "severity": "Severe",
        "treatment": "Remove infected plants. Disinfect tools. Control aphid population.",
        "prevention": "Use virus-free seeds. Wash hands before handling plants."
    },
    "Tomato_healthy": {
        "description": "The plant is perfectly healthy! No disease detected.",
        "severity": "None",
        "treatment": "No treatment needed.",
        "prevention": "Maintain proper watering, fertilization, and sunlight."
    }
}

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Helper functions
def get_severity_badge(severity):
    if severity == "None":
        return "✅ Healthy"
    elif severity == "Mild":
        return "🟢 Mild"
    elif severity == "Moderate":
        return "🟡 Moderate"
    else:
        return "🔴 Severe"

def predict(image):
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = round(float(np.max(prediction)) * 100, 2)
    return predicted_class, confidence

def generate_pdf(predicted_class, confidence, info):
    display_name = predicted_class.replace("_", " ").replace("  ", " ")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(45, 106, 79)
    pdf.cell(0, 15, "Plant Disease Detection Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Detection Result", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Disease: {display_name}", ln=True)
    pdf.cell(0, 8, f"Confidence: {confidence}%", ln=True)
    pdf.cell(0, 8, f"Severity: {info['severity']}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "About", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, info['description'])
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Treatment", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, info['treatment'])
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Prevention Tips", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, info['prevention'])
    return pdf.output(dest='S').encode('latin-1')

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🌿 Plant Doctor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Detect Disease", "📋 History"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("### 📊 Stats")
    st.metric("Total Scans", len(st.session_state.history))
    if st.session_state.history:
        diseases = [h['class'] for h in st.session_state.history if h['severity'] != 'None']
        st.metric("Diseases Found", len(diseases))
        healthy = [h for h in st.session_state.history if h['severity'] == 'None']
        st.metric("Healthy Plants", len(healthy))
    st.markdown("---")
    st.markdown("**Built by Muhilan G**")
    st.markdown("B.Tech CSE - SRM University")

# ── HOME PAGE ──
if page == "🏠 Home":
    st.markdown('<div class="title">🌿 Plant Disease Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered plant health analysis using Deep Learning</div>', unsafe_allow_html=True)
    st.write("")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="home-card">
            <h2>🧠</h2>
            <h4>Deep Learning</h4>
            <p>Powered by MobileNetV2 CNN trained on 54,000+ leaf images</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="home-card">
            <h2>⚡</h2>
            <h4>92% Accuracy</h4>
            <p>High accuracy detection across 15 plant disease classes</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="home-card">
            <h2>💊</h2>
            <h4>Treatment Advice</h4>
            <p>Get instant treatment and prevention tips for detected diseases</p>
        </div>""", unsafe_allow_html=True)

    st.write("")
    st.markdown("### 🌱 Supported Plants")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🫑 **Bell Pepper**\nBacterial Spot, Healthy")
    with c2:
        st.info("🥔 **Potato**\nEarly Blight, Late Blight, Healthy")
    with c3:
        st.info("🍅 **Tomato**\n9 diseases + Healthy")

    st.write("")
    st.markdown("### 🚀 How to Use")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="home-card"><h3>1️⃣</h3><p>Go to <b>Detect Disease</b> page</p></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="home-card"><h3>2️⃣</h3><p>Upload a <b>leaf image</b></p></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="home-card"><h3>3️⃣</h3><p>Get <b>instant results</b></p></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="home-card"><h3>4️⃣</h3><p>Download <b>PDF report</b></p></div>', unsafe_allow_html=True)

# ── DETECT PAGE ──
elif page == "🔍 Detect Disease":
    st.markdown('<div class="title">🔍 Detect Plant Disease</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload one or more leaf images for analysis</div>', unsafe_allow_html=True)
    st.write("")

    uploaded_files = st.file_uploader(
        "📤 Upload leaf images (you can select multiple)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            st.write("---")
            image = Image.open(uploaded_file).convert('RGB')
            predicted_class, confidence = predict(image)
            display_name = predicted_class.replace("_", " ").replace("  ", " ")
            info = disease_info.get(predicted_class, {
                "description": "Information not available.",
                "severity": "Unknown",
                "treatment": "Consult an expert.",
                "prevention": "Monitor the plant regularly."
            })

            # Save to history
            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "filename": uploaded_file.name,
                "class": predicted_class,
                "display": display_name,
                "confidence": confidence,
                "severity": info['severity']
            })

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=uploaded_file.name, use_column_width=True)
            with col2:
                st.markdown(f"""
                    <div class="result-box">
                        <h3>🔍 {display_name}</h3>
                        <p>Confidence: <b>{confidence}%</b></p>
                        <p>Severity: <b>{get_severity_badge(info['severity'])}</b></p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="info-box">
                        <b>📋 About:</b><br>{info['description']}
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="warning-box">
                        <b>💊 Treatment:</b><br>{info['treatment']}
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="result-box">
                        <b>🛡️ Prevention:</b><br>{info['prevention']}
                    </div>
                """, unsafe_allow_html=True)

                # PDF Download
                try:
                    pdf_bytes = generate_pdf(predicted_class, confidence, info)
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"report_{uploaded_file.name}.pdf",
                        mime="application/pdf"
                    )
                except Exception:
                    st.info("Install fpdf2 for PDF reports: pip install fpdf2")

# ── HISTORY PAGE ──
elif page == "📋 History":
    st.markdown('<div class="title">📋 Detection History</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">All your previous plant disease scans</div>', unsafe_allow_html=True)
    st.write("")

    if not st.session_state.history:
        st.info("No detections yet. Go to **Detect Disease** to scan your first leaf!")
    else:
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()

        for i, record in enumerate(reversed(st.session_state.history)):
            st.markdown(f"""
                <div class="history-card">
                    <b>#{len(st.session_state.history) - i} — {record['display']}</b><br>
                    📁 File: {record['filename']} &nbsp;|&nbsp;
                    💯 Confidence: {record['confidence']}% &nbsp;|&nbsp;
                    ⚠️ Severity: {get_severity_badge(record['severity'])} &nbsp;|&nbsp;
                    🕐 {record['time']}
                </div>
            """, unsafe_allow_html=True)
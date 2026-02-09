import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="AI² Brain Diagnostic Lab",
    page_icon="🧠",
    layout="centered"
)

# --- LOAD MODEL ---
@st.cache_resource
def load_medical_model():
    # Ensure the path matches your GitHub structure
    return tf.keras.models.load_model('models/max_val_brain_model.keras')

try:
    model = load_medical_model()
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
except Exception as e:
    st.error(f"Error loading model: {e}")

# --- UI INTERFACE ---
st.title("🧠 AI² Brain Tumor Detector")
st.markdown("### Applied Generative AI for Intelligent Applications")
st.write("Upload an MRI scan to generate an AI-powered diagnostic prediction.")

uploaded_file = st.file_uploader("Upload Image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 1. Display Image
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Target MRI Scan", use_column_width=True)
    
    # 2. Preprocessing (Must match training exactly)
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Crucial: This scales the pixels specifically for MobileNetV2
    img_preprocessed = preprocess_input(img_array)
    
    if st.button("Generate Analysis"):
        with st.spinner("Analyzing scan layers..."):
            predictions = model.predict(img_preprocessed)
            result_idx = np.argmax(predictions)
            result_label = class_names[result_idx]
            confidence = predictions[0][result_idx]

        # 3. Display Results
        st.divider()
        st.subheader(f"Diagnosis: {result_label}")
        st.write(f"**Confidence Score:** {confidence:.2%}")
        
        # UI Feedback based on result
        if result_label == "No Tumor":
            st.success("Result: Normal. No visible tumor detected in this scan.")
        else:
            st.error(f"Alert: Potential {result_label} detected. Please consult a radiologist.")
            
        st.info("**Disclaimer:** This tool is for educational purposes (AI² Course) and should not replace professional medical advice.")

st.markdown("---")
st.caption("Developed as part of the Value Added Course on AI²")

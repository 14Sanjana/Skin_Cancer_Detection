import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image
import os

# Load the trained model
model = load_model("skin_cancer_model.h5")

# Class labels
class_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# Risk level and precautions
risk_levels = {
    'MEL': 'High',
    'AKIEC': 'High',
    'BCC': 'Medium',
    'VASC': 'Medium',
    'NV': 'Low',
    'BKL': 'Low',
    'DF': 'Low'
}

precautions = {
    'MEL': "See a dermatologist immediately. Monitor moles for changes. Use SPF 50+.",
    'NV': "Usually benign. Monitor moles. Annual check-ups recommended.",
    'BCC': "Consult a dermatologist. Avoid sun exposure and use sunscreen.",
    'AKIEC': "Precancerous lesion – seek evaluation. Avoid UV rays.",
    'BKL': "Harmless but monitor for changes. Keep skin moisturized.",
    'DF': "Benign. No treatment needed unless growing or irritated.",
    'VASC': "Usually harmless. If it grows or bleeds, seek evaluation."
}

# Streamlit UI
st.set_page_config(page_title="Skin Cancer Detection", layout="centered")
st.title("🧪 Skin Cancer Detection App")
st.write("Upload a skin lesion image to detect its type, risk level, and suggested precautions.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

# If no file is uploaded, use default sample
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
else:
    st.warning("No image uploaded. Using default sample image.")
    sample_path = "sample.jpg"
    if not os.path.exists(sample_path):
        st.error("Sample image not found. Please upload one.")
        st.stop()
    img = Image.open(sample_path).convert("RGB")

# Display uploaded/default image
st.image(img, caption="Input Image", use_column_width=True)

# Predict button
if st.button("🔍 Predict"):
    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)
    idx = np.argmax(prediction)
    confidence = float(np.max(prediction))
    label = class_labels[idx]

    # Output results
    st.markdown(f"### 🧬 Predicted Class: **{label}**")
    st.markdown(f"### ✅ Confidence: **{confidence * 100:.2f}%**")
    st.markdown(f"### ⚠️ Risk Level: **{risk_levels[label]}**")
    st.markdown(f"### 🛡️ Precautions: {precautions[label]}")

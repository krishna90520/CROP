import os
import urllib.request
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms

# Disable Streamlit file watcher to prevent reload issues
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"

# Define class labels for each crop type
CLASS_LABELS = {
    "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
    "GroundNut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
    "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage", "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
}

# Model storage (Replace with actual cloud link)
MODEL_URLS = {
    "Paddy": "https://your-cloud-storage-link/classification_4Disease_best.pt",
    "GroundNut": "https://your-cloud-storage-link/groundnut_best.pt",
    "Cotton": "https://your-cloud-storage-link/cotton_best.pt"
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Download model if not available
def download_model(crop_type):
    model_path = os.path.join(MODEL_DIR, f"{crop_type}.pt")
    if not os.path.exists(model_path):
        st.info(f"Downloading {crop_type} model...")
        urllib.request.urlretrieve(MODEL_URLS[crop_type], model_path)
    return model_path

# Load the YOLO classification model
@st.cache_resource
def load_model(crop_type):
    try:
        model_path = download_model(crop_type)
        model = YOLO(model_path)  # Load YOLO model
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((416, 416)),  # Resize for YOLO
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Classification function
def classify_image(image, crop_type):
    model = load_model(crop_type)
    if model is None:
        return None

    image_tensor = preprocess_image(image)

    with torch.no_grad():
        results = model(image_tensor)  # Run classification
        output = results[0].probs.data.cpu().numpy()  # Get probability scores

    # Get predicted class index
    predicted_idx = np.argmax(output)
    predicted_label = CLASS_LABELS[crop_type][predicted_idx]

    return predicted_label, output.tolist()

# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Crop Disease Classification</h1>", unsafe_allow_html=True)

st.markdown("### Select the Crop Type")
crop_selection = st.selectbox("Select the crop", ["Paddy", "GroundNut", "Cotton"], label_visibility="hidden")
st.write(f"Selected Crop: {crop_selection}")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Classify Disease"):
        with st.spinner("Classifying..."):
            result = classify_image(img, crop_selection)

            if result:
                predicted_label, probabilities = result
                st.success(f"Predicted Disease: {predicted_label}")
                st.write(f"Confidence Scores: {probabilities}")
            else:
                st.error("Error in classification.")

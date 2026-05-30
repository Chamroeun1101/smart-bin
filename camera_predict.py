import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("models/smart_bin_model_v2.h5")

# Title
st.title("♻️ Smart Bin Waste Classification")

st.write("Upload an image or use your camera to classify waste.")

# -----------------------------
# FUNCTION FOR PREDICTION
# -----------------------------
def predict_waste(img):

    # Resize image
    img = img.resize((224, 224))

    # Convert to array
    img_array = image.img_to_array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Classification
    if prediction[0][0] > 0.5:
        label = "Plastic"
        confidence = prediction[0][0] * 100
    else:
        label = "Biological"
        confidence = (1 - prediction[0][0]) * 100

    return label, confidence


# ====================================
# IMAGE UPLOAD SECTION
# ====================================

uploaded_file = st.file_uploader(
    "Upload Waste Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image", use_container_width=True)

    label, confidence = predict_waste(img)

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")


# ====================================
# CAMERA SECTION
# ====================================

st.subheader("📷 Camera Prediction")

camera_image = st.camera_input("Take a picture")

if camera_image is not None:

    img = Image.open(camera_image)

    st.image(img, caption="Captured Image", use_container_width=True)

    label, confidence = predict_waste(img)

    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")
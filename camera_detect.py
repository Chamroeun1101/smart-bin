import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load model
model = load_model("models/smart_bin_model_v2.h5")

# Title
st.title("Smart Bin Classification System")

st.write("Upload an image to classify waste.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    img = Image.open(uploaded_file)

    # Show image
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Resize image
    img = img.resize((224, 224))

    # Convert to array
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Classify
    if prediction[0][0] > 0.5:
        label = "Plastic"
        confidence = prediction[0][0] * 100
    else:
        label = "Biological"
        confidence = (1 - prediction[0][0]) * 100

    # Show result
    st.success(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")
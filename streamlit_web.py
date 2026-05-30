import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing import image    # type: ignore
import numpy as np
import pandas as pd
from PIL import Image
import time


# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Smart Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

model = load_model("models/smart_bin_model_v2.h5")

# =====================================
# SESSION STATE
# =====================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_waste(img):

    img = img.resize((224, 224))

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        label = "Plastic"
        confidence = prediction[0][0] * 100
    else:
        label = "Biological"
        confidence = (1 - prediction[0][0]) * 100

    return label, confidence

# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("♻️ Navigation")

menu = st.sidebar.radio(
    "Go to",
    ["📷 Prediction", "📊 Dashboard"]
)

st.sidebar.markdown("---")

st.sidebar.success("AI Waste Detection System")

# =====================================
# PREDICTION PAGE
# =====================================

if menu == "📷 Prediction":

    st.title("📷 Smart Waste Prediction")

    st.write(
        "Upload a waste image or use your camera "
        "to classify waste using AI."
    )

    st.markdown("---")

    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])

    # =================================
    # UPLOAD TAB
    # =================================

    with tab1:

        uploaded_file = st.file_uploader(
            "Upload Waste Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            img = Image.open(uploaded_file)

            with st.spinner("🤖 AI is analyzing image..."):
                time.sleep(1)

                label, confidence = predict_waste(img)

            st.markdown("---")

            col1, col2 = st.columns(2)

            # =========================
            # IMAGE DISPLAY
            # =========================

            with col1:

                st.image(
                    img,
                    caption="Uploaded Image",
                    use_container_width=True
                )

            # =========================
            # RESULT DISPLAY
            # =========================

            with col2:

                st.subheader("♻️ Prediction Result")

                if label == "Plastic":

                    st.success(f"Prediction: {label}")

                    waste_type = "Recyclable Waste"

                    suggestion = (
                        "Dispose in the recycling bin."
                    )

                else:

                    st.warning(f"Prediction: {label}")

                    waste_type = "Organic Waste"

                    suggestion = (
                        "Suitable for composting."
                    )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(float(confidence / 100))

                st.info(f"Category: {waste_type}")

                st.subheader("🌱 Recycling Suggestion")

                st.write(suggestion)

            # =========================
            # SAVE HISTORY
            # =========================

            st.session_state.history.append({
                "Type": label,
                "Confidence": round(confidence, 2)
            })

    # =================================
    # CAMERA TAB
    # =================================

    with tab2:

        camera_image = st.camera_input(
            "Take a picture"
        )

        if camera_image is not None:

            img = Image.open(camera_image)

            with st.spinner("🤖 AI is analyzing image..."):
                time.sleep(1)

                label, confidence = predict_waste(img)

            st.markdown("---")

            col1, col2 = st.columns(2)

            # =========================
            # IMAGE DISPLAY
            # =========================

            with col1:

                st.image(
                    img,
                    caption="Captured Image",
                    use_container_width=True
                )

            # =========================
            # RESULT DISPLAY
            # =========================

            with col2:

                st.subheader("♻️ Prediction Result")

                if label == "Plastic":

                    st.success(f"Prediction: {label}")

                    waste_type = "Recyclable Waste"

                    suggestion = (
                        "Dispose in the recycling bin."
                    )

                else:

                    st.warning(f"Prediction: {label}")

                    waste_type = "Organic Waste"

                    suggestion = (
                        "Suitable for composting."
                    )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(float(confidence / 100))

                st.info(f"Category: {waste_type}")

                st.subheader("🌱 Recycling Suggestion")

                st.write(suggestion)

            # =========================
            # SAVE HISTORY
            # =========================

            st.session_state.history.append({
                "Type": label,
                "Confidence": round(confidence, 2)
            })

# =====================================
# DASHBOARD PAGE
# =====================================

elif menu == "📊 Dashboard":

    st.title("📊 Detection Dashboard")

    st.write(
        "Visualization of previous AI predictions."
    )

    st.markdown("---")

    history = st.session_state.history

    # =================================
    # EMPTY HISTORY
    # =================================

    if len(history) == 0:

        st.warning(
            "No prediction history available yet."
        )

    # =================================
    # SHOW DASHBOARD
    # =================================

    else:

        df = pd.DataFrame(history)

        # =============================
        # TABLE
        # =============================

        st.subheader("📋 Prediction History")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.markdown("---")

        # =============================
        # STATISTICS
        # =============================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📈 Waste Detection Count")

            count_data = df["Type"].value_counts()

            st.bar_chart(count_data)

        with col2:

            st.subheader("📊 Confidence Distribution")

            st.line_chart(df["Confidence"])

        st.markdown("---")

        # =============================
        # SUMMARY METRICS
        # =============================

        st.subheader("📌 Summary")

        total_predictions = len(df)

        avg_confidence = df["Confidence"].mean()

        plastic_count = (
            df[df["Type"] == "Plastic"].shape[0]
        )

        biological_count = (
            df[df["Type"] == "Biological"].shape[0]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Predictions",
            total_predictions
        )

        c2.metric(
            "Average Confidence",
            f"{avg_confidence:.2f}%"
        )

        c3.metric(
            "Plastic Detected",
            plastic_count
        )

        c4.metric(
            "Biological Detected",
            biological_count
        )


import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load trained model
model = load_model("models/smart_bin_model_v2.h5")

# class labels
classes = ["Biological", "Plastic"]

# Open laptop camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Resize frame for model
    img = cv2.resize(frame, (224, 224))

    # Normalize image
    img_array = img / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    # Binary classification
    if prediction[0][0] > 0.5:
        label = "Plastic"
    else:
        label = "Biological"

    # Show label on screen
    cv2.putText(
        frame,
        f'Prediction: {label}',
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show camera window
    cv2.imshow("Smart Bin System", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
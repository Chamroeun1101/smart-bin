from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = load_model("models/smart_bin_model_v2.h5")

# Image path
img_path = "test_images/flosh.jpg"

# Load and resize image
img = image.load_img(img_path, target_size=(224, 224))

# Convert to array
img_array = image.img_to_array(img)

# Normalize
img_array = img_array / 255.0

# Add batch dimension
img_array = np.expand_dims(img_array, axis=0)

# Prediction
prediction = model.predict(img_array)

# Binary classification
if prediction[0][0] > 0.5:
    label = "Plastic"
else:
    label = "Biological"

# Show image
plt.imshow(img)
plt.title(f"Prediction: {label}")
plt.axis("off")
plt.show()

print("Prediction:", label)
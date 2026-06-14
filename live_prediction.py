import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────
# Load saved model
# ─────────────────────────────────────────
model = load_model('models/smart_bin_model_v2.h5')
print('✅ Model loaded!')

# ─────────────────────────────────────────
# Settings
# ─────────────────────────────────────────
# 0 = laptop camera
# 1 = phone camera via DroidCam USB
CAMERA_INDEX = 1
IMG_SIZE     = (224, 224)
CLASSES      = ['Biological', 'Plastic']

# Colors
GREEN  = (50,  205, 50)
BLUE   = (255, 100,  0)
WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)
RED    = (0,   0,   255)

# ─────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────
def predict(frame):
    img  = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img  = img.resize(IMG_SIZE)
    arr  = np.array(img) / 255.0
    arr  = np.expand_dims(arr, axis=0)
    prob = model.predict(arr, verbose=0)[0][0]

    if prob > 0.5:
        label = 'Plastic'
        conf  = prob * 100
        color = BLUE
    else:
        label = 'Biological'
        conf  = (1 - prob) * 100
        color = GREEN

    return label, conf, color

# ─────────────────────────────────────────
# Open camera
# ─────────────────────────────────────────
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print('❌ Cannot open camera!')
    print('   Try changing CAMERA_INDEX to 0')
    exit()

print('✅ Camera opened!')
print('   Press Q to quit')
print('   Press S to save screenshot')

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print('❌ Cannot read frame!')
        break

    # Predict every 10 frames (for speed)
    if frame_count % 10 == 0:
        label, conf, color = predict(frame)

    frame_count += 1

    # ─────────────────────────────────────
    # Draw UI on frame
    # ─────────────────────────────────────
    h, w = frame.shape[:2]

    # Top banner background
    cv2.rectangle(frame, (0, 0), (w, 70), BLACK, -1)

    # Title
    cv2.putText(
        frame, 'Smart Bin System',
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
        0.7, WHITE, 2
    )

    # Prediction label
    cv2.putText(
        frame, f'Prediction: {label}',
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, color, 2
    )

    # Confidence on right side
    cv2.putText(
        frame, f'Confidence: {conf:.1f}%',
        (w - 250, 55), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, WHITE, 2
    )

    # Bottom banner
    cv2.rectangle(frame, (0, h - 40), (w, h), BLACK, -1)

    # Bottom instruction
    cv2.putText(
        frame, 'Press Q to quit  |  Press S to save',
        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
        0.5, WHITE, 1
    )

    # Confidence bar
    bar_width = int((conf / 100) * (w - 20))
    cv2.rectangle(frame, (10, h - 55), (w - 10, h - 45), (50, 50, 50), -1)
    cv2.rectangle(frame, (10, h - 55), (10 + bar_width, h - 45), color, -1)

    # Border color based on prediction
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 4)

    # Show frame
    cv2.imshow('Smart Bin — Live Prediction', frame)

    # Key controls
    key = cv2.waitKey(1) & 0xFF

    # Q = quit
    if key == ord('q'):
        print('Closing...')
        break

    # S = save screenshot
    if key == ord('s'):
        filename = f'screenshot_{frame_count}.jpg'
        cv2.imwrite(filename, frame)
        print(f'✅ Saved: {filename}')

# ─────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
print('Done!')
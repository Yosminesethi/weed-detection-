import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load model
model = load_model("weed_model.h5")

# Load image
img = cv2.imread("test.jpg")
img = cv2.resize(img, (70, 70))
img = img / 255.0
img = np.reshape(img, (1, 70, 70, 3))

# 🔥 THIS LINE WAS MISSING
prediction = model.predict(img)

# Class labels
classes = [
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet"
]

# Get result
predicted_class = np.argmax(prediction)

print("Predicted plant:", classes[predicted_class])
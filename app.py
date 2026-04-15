from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os
import uuid
import base64

# YOLO
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔥 Load models
cnn_model = load_model("weed_model.h5")
yolo_model = YOLO("yolov8n.pt")

# Classes
classes = [
    "Black-grass", "Charlock", "Cleavers", "Common Chickweed",
    "Common wheat", "Fat Hen", "Loose Silky-bent",
    "Maize", "Scentless Mayweed", "Shepherds Purse",
    "Small-flowered Cranesbill", "Sugar beet"
]

crops = ["Common wheat", "Maize", "Sugar beet"]


# 🔥 YOLO detection function
def detect_weeds(image_path):
    results = yolo_model(image_path)

    for r in results:
        img = r.plot()  # draw bounding boxes

        output_path = os.path.join(
            UPLOAD_FOLDER, str(uuid.uuid4()) + ".jpg"
        )
        cv2.imwrite(output_path, img)

    return output_path


# 🔥 MAIN PAGE
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    action = ""
    image_path = ""

    if request.method == "POST":
        file = request.files["file"]

        filename = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(image_path)

        img = cv2.imread(image_path)

        if img is None:
            return "Invalid image"

        # 🔹 CNN Prediction
        img_resized = cv2.resize(img, (70, 70)) / 255.0
        img_resized = np.reshape(img_resized, (1, 70, 70, 3))

        pred = cnn_model.predict(img_resized)[0]
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class] * 100

        plant = classes[predicted_class]

        # 🔹 Decision logic
        if confidence < 60:
            action = "⚠️ Low confidence prediction"
        elif plant in crops:
            action = "✅ This is a healthy crop"
        else:
            action = "⚠️ This appears to be a weed"

        prediction = f"{plant} — {confidence:.2f}% confidence"

        # 🔥 YOLO bounding boxes
        image_path = detect_weeds(image_path)

    return render_template(
        "index.html",
        prediction=prediction,
        action=action,
        image_path=image_path
    )


# 🔥 CAMERA CAPTURE ROUTE
@app.route("/capture", methods=["POST"])
def capture():
    data = request.get_json()
    image_data = data["image"]

    # Decode base64 image
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    filename = str(uuid.uuid4()) + ".jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # 🔹 Run CNN
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (70, 70)) / 255.0
    img_resized = np.reshape(img_resized, (1, 70, 70, 3))

    pred = cnn_model.predict(img_resized)[0]
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class] * 100
    plant = classes[predicted_class]

    # 🔹 YOLO bounding boxes
    output_path = detect_weeds(image_path)

    return jsonify({
        "image_path": output_path,
        "prediction": f"{plant} ({confidence:.2f}%)"
    })


# 🔥 RUN APP (DEPLOY READY)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
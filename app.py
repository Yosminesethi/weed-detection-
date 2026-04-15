from flask import Flask, render_template, request, jsonify
import os
import uuid
import base64
import cv2

# YOLO
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔥 Load YOLO model
yolo_model = YOLO("yolov8n.pt")


# 🔥 YOLO detection function
def detect_weeds(image_path):
    results = yolo_model(image_path)

    label = "No weed detected"
    action = "✅ Healthy crop"

    for r in results:
        img = r.plot()  # bounding boxes

        if len(r.boxes) > 0:
            label = "Weed detected"
            action = "⚠️ Weed detected"

        output_path = os.path.join(
            UPLOAD_FOLDER, str(uuid.uuid4()) + ".jpg"
        )
        cv2.imwrite(output_path, img)

    return output_path, label, action


# 🔥 MAIN PAGE
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    action = ""
    image_path = ""

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = str(uuid.uuid4()) + ".jpg"
            image_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(image_path)

            # 🔥 YOLO detection
            image_path, prediction, action = detect_weeds(image_path)

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

    # Decode base64
    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    filename = str(uuid.uuid4()) + ".jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    # 🔥 YOLO detection
    output_path, prediction, action = detect_weeds(image_path)

    return jsonify({
        "image_path": output_path,
        "prediction": prediction,
        "action": action
    })


# 🔥 RUN APP (IMPORTANT FOR RENDER)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
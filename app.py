from flask import Flask, render_template, request, jsonify
import os
import uuid
import base64
import cv2
from ultralytics import YOLO

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")


# 🔥 YOLO detection
def detect_weeds(image_path):
    results = model(image_path)

    label = "No weed detected"
    action = "✅ Healthy crop"

    for r in results:
        img = r.plot()

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

            image_path, prediction, action = detect_weeds(image_path)

    return render_template(
        "index.html",
        prediction=prediction,
        action=action,
        image_path=image_path
    )


# 🔥 CAMERA API
@app.route("/capture", methods=["POST"])
def capture():
    data = request.get_json()
    image_data = data["image"]

    image_data = image_data.split(",")[1]
    image_bytes = base64.b64decode(image_data)

    filename = str(uuid.uuid4()) + ".jpg"
    image_path = os.path.join(UPLOAD_FOLDER, filename)

    with open(image_path, "wb") as f:
        f.write(image_bytes)

    output_path, prediction, action = detect_weeds(image_path)

    return jsonify({
        "image_path": output_path,
        "prediction": prediction,
        "action": action
    })


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
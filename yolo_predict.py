from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

def detect_weeds(image_path):
    results = model(image_path)[0]

    img = cv2.imread(image_path)

    label = "No weed detected"
    action = "✅ Healthy crop"

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        text = f"{name} ({conf:.2f})"
        cv2.putText(img, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        label = name
        action = "⚠️ Weed detected"

    cv2.imwrite(image_path, img)

    return {
        "label": label,
        "action": action
    }
from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8n.pt")

def detect_weeds(image_path):
    results = model(image_path)

    for r in results:
        img = r.plot()

    output_path = os.path.join("static/uploads", "output.jpg")
    cv2.imwrite(output_path, img)

    return output_path
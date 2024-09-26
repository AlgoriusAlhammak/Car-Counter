from ultralytics import YOLO
import cv2
# Load the YOLOv8 model
model = YOLO('../Yolo-Weights/yolov8n.pt')

# Perform inference and display the results
results = model.predict(source="images/motobikes.png", show=True)

cv2.waitKey(0)

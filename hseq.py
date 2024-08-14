*** Taken from Chat GPT*** PROMPT: how can I fine-tune the model on a custom dataset containing PPE images?



import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image

# Load YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define the detection function using YOLOv5
def detect_objects(frame, model):
    # Convert frame to PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Perform inference
    results = model(img)
    
    # Parse results
    detections = results.xyxy[0].numpy()  # bounding box coordinates, confidence, and class
    
    # Draw bounding boxes and labels on the frame
    for detection in detections:
        bbox = detection[:4]  # x_min, y_min, x_max, y_max
        confidence = detection[4]
        class_id = int(detection[5])
        label = model.names[class_id]

        # Draw bounding box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Initialize Streamlit app
st.title("Real-Time Object Detection")

# Create a sidebar for camera options
camera_option = st.sidebar.selectbox("Select Camera", ["IP Cam", "Webcam"])
if camera_option == "IP Cam":
    camera_url = st.sidebar.text_input("Enter IP Camera URL", "http://your-ip-camera-url")
else:
    camera_index = st.sidebar.number_input("Enter Webcam Index", value=0, step=1)

# Streamlit video stream
frame_placeholder = st.empty()

# Capture video from the camera
if camera_option == "IP Cam":
    cap = cv2.VideoCapture(camera_url)
else:
    cap = cv2.VideoCapture(camera_index)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame.")
        break

    # Perform object detection on the frame
    frame = detect_objects(frame, model)

    # Display the frame in the Streamlit app
    frame_placeholder.image(frame, channels="BGR")

cap.release()

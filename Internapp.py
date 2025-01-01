import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

# Function to detect cars in an image
def detect_cars(image, model_path="yolov8n.pt"):
    """
    Detects cars in an image and returns cropped car images.
    """
    model = YOLO(model_path)
    results = model(image)
    cropped_cars = []

    for detection in results[0].boxes:
        class_id = int(detection.cls)
        label = model.names[class_id]

        if label.lower() == "car":  # Only detect cars
            x1, y1, x2, y2 = map(int, detection.xyxy[0])  # Bounding box coordinates
            cropped_car = image[y1:y2, x1:x2]  # Crop car region
            cropped_cars.append(cropped_car)

    return cropped_cars

# Function to detect car color
def detect_car_color(car_image):
    """
    Detects the dominant color of a car in the input image.
    """
    hsv_image = cv2.cvtColor(car_image, cv2.COLOR_BGR2HSV)

    # Define color ranges
    color_ranges = {
        "Red": ([136, 87, 111], [180, 255, 255]),
        "Green": ([25, 52, 72], [102, 255, 255]),
        "Blue": ([94, 80, 2], [120, 255, 255]),
    }

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower, np.uint8)
        upper = np.array(upper, np.uint8)
        mask = cv2.inRange(hsv_image, lower, upper)

        if cv2.countNonZero(mask) > 1000:  # Threshold for detection
            return color

    return "Unknown"

# Streamlit App
st.title("Car Detection and Color Identification")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Detect cars
    model_path = "yolov8n.pt"  # Ensure the YOLO weights are in the same directory
    car_images = detect_cars(image, model_path)

    if car_images:
        st.write(f"Detected {len(car_images)} car(s).")
        for i, car_image in enumerate(car_images):
            # Detect color
            car_color = detect_car_color(car_image)

            # Display cropped car and color
            st.image(cv2.cvtColor(car_image, cv2.COLOR_BGR2RGB), caption=f"Car {i + 1}")
            st.write(f"Car {i + 1} Color: {car_color}")
    else:
        st.write("No cars detected in the image.")

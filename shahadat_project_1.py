import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import warnings
import urllib.request

# Set page configuration
st.set_page_config(page_title="My YOLO11 Portfolio", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Me", "YOLO11 Project", "Object Detection App"])

# Load YOLO11 model
@st.cache_resource
def load_model():
    # Check if model exists, if not download it
    model_path = "yolo11n.pt"
    if not os.path.exists(model_path):
        with st.spinner("Downloading YOLO model... This might take a minute."):
            urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt", model_path)
    return YOLO(model_path)

model = load_model()

# About Me Section
if page == "About Me":
    st.title("Welcome to My Portfolio")
    col1, col2 = st.columns([1, 2])
    with col1:
        # Use a placeholder image if profile.jpg doesn't exist
        st.image("https://via.placeholder.com/200x200.png?text=Profile+Photo", width=200)
    with col2:
        st.header("About Me")
        st.write("""
        Hi! I'm [Your Name], a passionate data scientist and machine learning enthusiast.
        I specialize in computer vision and have experience with state-of-the-art models like YOLO11.
        This portfolio showcases my project using YOLO11 for object detection, built with Streamlit.
        Connect with me on [LinkedIn](https://linkedin.com/in/your-profile) or check out my [GitHub](https://github.com/your-username).
        """)

# YOLO11 Project Description
elif page == "YOLO11 Project":
    st.title("YOLO11 Object Detection Project")
    st.write("""
    ### Project Overview
    This project demonstrates object detection using the YOLO11 model pretrained on the COCO dataset.
    The app allows users to upload images and view detected objects with bounding boxes and labels.
    
    ### Technologies Used
    - **YOLO11**: For object detection (Ultralytics)
    - **Streamlit**: For the web interface
    - **Python**: Core programming
    - **PIL**: Image processing
    
    ### How It Works
    Upload an image in the 'Object Detection App' section, and the YOLO11 model will process it to detect objects.
    The results are displayed with annotated bounding boxes.
    """)

# Object Detection App
elif page == "Object Detection App":
    st.title("YOLO11 Object Detection App")
    st.write("Upload an image to detect objects using the pretrained YOLO11 model.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process image with YOLO11
        with st.spinner("Detecting objects..."):
            results = model(image)
            
            # Save and display the result
            result_image = results[0].plot()  # Get annotated image with boxes and labels
            result_image_pil = Image.fromarray(result_image)
            st.image(result_image_pil, caption="Detected Objects", use_container_width=True)
            
            # Optional: Show detection results
            st.write("**Detection Results**:")
            for det in results[0].boxes:
                label = results[0].names[int(det.cls)]
                conf = det.conf.item()
                st.write(f"- {label}: {conf:.2f} confidence")
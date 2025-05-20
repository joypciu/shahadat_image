import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import urllib.request
import torch

# Set page configuration
st.set_page_config(page_title="YOLO Object Detection Portfolio", layout="wide")

# Debug Streamlit version
st.write(f"Streamlit version: {st.__version__}")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Me", "YOLO Project", "Object Detection App"])

# Load YOLO model
@st.cache_resource
def load_model():
    # Allowlist the DetectionModel class for safe loading
    torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
    
    # Check if model exists, if not download it
    model_path = "yolov11n.pt"
    if not os.path.exists(model_path):
        with st.spinner("Downloading YOLOv11n model... This might take a minute."):
            urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt", model_path)
            
    return YOLO(model_path)

# Try to load model with error handling
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.info("Continuing without model functionality. Please try reloading the page.")
    model = None

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
        I specialize in computer vision and have experience with state-of-the-art models like YOLO.
        This portfolio showcases my project using YOLOv11 for object detection, built with Streamlit.
        Connect with me on [LinkedIn](https://linkedin.com/in/your-profile) or check out my [GitHub](https://github.com/your-username).
        """)

# YOLO Project Description
elif page == "YOLO Project":
    st.title("YOLOv11 Object Detection Project")
    st.write("""
    ### Project Overview
    This project demonstrates object detection using the YOLOv11n model pretrained on the COCO dataset.
    The app allows users to upload images and view detected objects with bounding boxes and labels.
    
    ### Technologies Used
    - **YOLOv11n**: For object detection (Ultralytics)
    - **Streamlit**: For the web interface
    - **Python**: Core programming
    - **PIL**: Image processing
    
    ### How It Works
    Upload an image in the 'Object Detection App' section, and the YOLOv11n model will process it to detect objects.
    The results are displayed with annotated bounding boxes.
    """)

# Object Detection App
elif page == "Object Detection App":
    st.title("YOLOv11 Object Detection App")
    st.write("Upload an image to detect objects using the pretrained YOLOv11n model.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and display the uploaded image
        image = Image.open(uploaded_file)
        # Conditionally use use_container_width based on Streamlit version
        try:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        except TypeError:
            st.image(image, caption="Uploaded Image", width=None)  # Fallback for older Streamlit versions
        
        # Process image with YOLO
        if model is not None:
            try:
                with st.spinner("Detecting objects..."):
                    results = model(image)
                    
                    # Save and display the result
                    result_image = results[0].plot()  # Get annotated image with boxes and labels
                    result_image_pil = Image.fromarray(result_image)
                    # Conditionally use use_container_width for result image
                    try:
                        st.image(result_image_pil, caption="Detected Objects", use_container_width=True)
                    except TypeError:
                        st.image(result_image_pil, caption="Detected Objects", width=None)  # Fallback for older Streamlit versions
                    
                    # Show detection results
                    st.write("**Detection Results:**")
                    for det in results[0].boxes:
                        label = results[0].names[int(det.cls)]
                        conf = det.conf.item()
                        st.write(f"- {label}: {conf:.2f} confidence")
            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.info("Please try with another image or reload the page.")
        else:
            st.warning("Model not loaded. Please try reloading the page.")
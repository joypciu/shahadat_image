import streamlit as st
from PIL import Image
import os

# Set page configuration
st.set_page_config(page_title="Computer Vision Portfolio", layout="wide")

# Check deployment type and show appropriate message
deployment_type = "Streamlit Cloud"
model_available = False

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["About Me", "YOLO Project", "Object Detection Demo"])

# About Me Section
if page == "About Me":
    st.title("Welcome to My Portfolio")
    col1, col2 = st.columns([1, 2])
    with col1:
        # Use a placeholder image
        st.image("https://via.placeholder.com/200x200.png?text=Profile+Photo", width=200)
    with col2:
        st.header("About Me")
        st.write("""
        Hi! I'm [Your Name], a passionate data scientist and machine learning enthusiast.
        I specialize in computer vision and have experience with state-of-the-art models like YOLO.
        This portfolio showcases my project using YOLO for object detection, built with Streamlit.
        Connect with me on [LinkedIn](https://linkedin.com/in/your-profile) or check out my [GitHub](https://github.com/your-username).
        """)

# YOLO Project Description
elif page == "YOLO Project":
    st.title("YOLO Object Detection Project")
    st.write("""
    ### Project Overview
    This project demonstrates object detection using the YOLOv8 model pretrained on the COCO dataset.
    The app allows users to view sample object detection results.
    
    ### Technologies Used
    - **YOLOv8**: For object detection (Ultralytics)
    - **Streamlit**: For the web interface
    - **Python**: Core programming
    - **PIL**: Image processing
    
    ### How YOLO Works
    YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system. 
    It works by dividing the image into a grid and predicting bounding boxes and class 
    probabilities for each cell in the grid, all in a single network pass.
    """)
    
    # Show sample YOLO detection images
    st.subheader("Sample Detection Results")
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://ultralytics.com/images/zidane.jpg", caption="Sample input image")
    with col2:
        st.image("https://ultralytics.com/images/bus.jpg", caption="Sample detection result")

# Object Detection Demo
elif page == "Object Detection Demo":
    st.title("Object Detection Demo")
    
    if model_available:
        st.write("Upload an image to detect objects using the pretrained YOLO model.")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Load and display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            st.error("Live object detection is currently unavailable in this demo version.")
            st.info("Please check back later for the full implementation.")
    else:
        st.warning("⚠️ This is a demo version without the actual YOLO model integration.")
        st.info("""
        In the full version, this app lets you upload images and get real-time object detection results.
        
        ### Example of what the detection looks like:
        """)
        
        # Show sample input/output
        col1, col2 = st.columns(2)
        with col1:
            st.image("https://ultralytics.com/images/zidane.jpg", caption="Example input image")
        with col2:
            st.image("https://ultralytics.com/images/zidane_result.jpg", caption="Example detection result")
        
        st.write("""
        ### Sample Detection Results:
        - Person: 0.94 confidence
        - Tie: 0.88 confidence
        - Person: 0.87 confidence
        """)
        
# Footer
st.markdown("---")
st.markdown("© 2025 | Created with Streamlit | [GitHub Repository](https://github.com/your-username/yolo-object-detection-app)")
import streamlit as st
from pdf2image import convert_from_path
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
import json

# Initialize Streamlit app
st.title("OCR Application")

# Sidebar for entering Google Cloud Vision API key
st.sidebar.title("Google Cloud Vision API Configuration")
json_key = st.sidebar.text_area("Enter your Google Cloud Vision API key (JSON format)", height=300)

# Try to import OpenCV and handle import errors
try:
    import cv2
except ImportError as e:
    st.error("Failed to import OpenCV. Please make sure it is installed correctly.")
    st.stop()

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def get_client_from_json(json_str):
    try:
        credentials = service_account.Credentials.from_service_account_info(json_str)
        return vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        st.sidebar.error(f"Failed to create client: {e}")
        return None

if json_key:
    try:
        # Convert the JSON key string to a dictionary
        key_dict = json.loads(json_key)
        
        # Initialize Google Cloud Vision client
        client = get_client_from_json(key_dict)
        
        if client and uploaded_file is not None:
            # Save uploaded file to disk
            with open("uploaded_file.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Convert PDF to images
            images = convert_from_path("uploaded_file.pdf", poppler_path="/usr/bin")
            img = np.array(images[0])  # Convert the first page to a numpy array

            # Define coordinates for the region containing customer information
            top = 200  # Define the top of the crop area
            bottom = 600  # Define the bottom of the crop area
            left = 20  # Define the left side of the crop area
            right = 3000  # Define the right side of the crop area

            # Crop the image to keep only the region with customer information
            img_cropped = img[top:bottom, left:right]

            # Convert to grayscale
            img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

            # OCR with Google Cloud Vision
            success, img_jpg = cv2.imencode('.jpg', img_gray)
            byte_img = img_jpg.tobytes()
            google_img = vision.Image(content=byte_img)
            resp = client.text_detection(image=google_img)
            if resp.text_annotations:
                ocr_output = resp.text_annotations[0].description.replace('\n', ' ')
            else:
                ocr_output = "No text detected"

            # Display OCR result
            st.write(ocr_output)
            
            # Optionally, show cropped image
            st.image(img_cropped, use_column_width=True)
    except json.JSONDecodeError:
        st.sidebar.error("Invalid JSON key. Please check your input.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.sidebar.warning("Please enter your Google Cloud Vision API key.")

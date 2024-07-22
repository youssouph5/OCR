import streamlit as st
from pdf2image import convert_from_path
import cv2
import numpy as np
from google.cloud import vision

# Streamlit app title and file uploader
st.title("OCR Application")
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Define Google Cloud Vision client configuration
client = vision.ImageAnnotatorClient.from_service_account_json('./ocrdemo-2024-fe6fecb70912.json')

def process_pdf(file_path):
    # Convert PDF to images
    images = convert_from_path(file_path, poppler_path="/usr/bin")
    img = np.array(images[0])  # Convert the first page to a numpy array

    # Define coordinates for the region containing customer information
    top = 468  # Define the top of the crop area
    bottom = 840  # Define the bottom of the crop area
    left = 50  # Define the left side of the crop area
    right = 3000  # Define the right side of the crop area

    # Crop the image to keep only the region with customer information
    img_cropped = img[top:bottom, left:right]

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)

    return img_gray

def perform_ocr(img_gray):
    # OCR with Google Cloud Vision
    success, img_jpg = cv2.imencode('.jpg', img_gray)
    byte_img = img_jpg.tobytes()
    google_img = vision.Image(content=byte_img)
    resp = client.text_detection(image=google_img)
    ocr_output = resp.text_annotations[0].description.replace('\n', ' ')
    
    return ocr_output

if uploaded_file is not None:
    # Save the uploaded file to disk
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process the PDF to get the image
    img_gray = process_pdf("uploaded_file.pdf")

    # Perform OCR on the image
    ocr_output = perform_ocr(img_gray)

    # Display OCR result
    st.write(ocr_output)
    
    # Optionally, show cropped image
    st.image(img_gray, use_column_width=True)

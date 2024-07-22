import streamlit as st
import fitz  # PyMuPDF
import cv2
import numpy as np
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import json

# Initialize Streamlit app
st.title("OCR Application")

# Sidebar for uploading Google Cloud Vision API JSON key file
st.sidebar.title("Google Cloud Vision API Configuration")
json_file = st.sidebar.file_uploader("Upload your Google Cloud Vision API key file (JSON)", type=["json"])

# Upload PDF file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

def get_client_from_json_file(json_file):
    try:
        key_dict = json.load(json_file)
        credentials = service_account.Credentials.from_service_account_info(key_dict)
        return vision.ImageAnnotatorClient(credentials=credentials)
    except Exception as e:
        st.sidebar.error(f"Failed to create client: {e}")
        return None

def convert_pdf_to_image(file_path):
    # Open the PDF file
    doc = fitz.open(file_path)
    # Select the first page
    page = doc.load_page(0)
    # Convert the page to an image
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(img)

if json_file is not None:
    client = get_client_from_json_file(json_file)
    
    if client and uploaded_file is not None:
        # Save uploaded file to disk
        with open("uploaded_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Convert PDF to images
        img = convert_pdf_to_image("uploaded_file.pdf")

        # Display the image and allow user to draw a region
        st.image(img, caption="Original Image", use_column_width=True)

        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fill color with some transparency
            stroke_width=2,
            stroke_color="#ff0000",
            background_image=Image.fromarray(img),
            update_streamlit=True,
            height=img.shape[0],
            width=img.shape[1],
            drawing_mode="rect",
            key="canvas",
        )

        if canvas_result.json_data is not None:
            for shape in canvas_result.json_data["objects"]:
                if shape["type"] == "rect":
                    left = int(shape["left"])
                    top = int(shape["top"])
                    width = int(shape["width"])
                    height = int(shape["height"])
                    right = left + width
                    bottom = top + height

                    # Crop the image to the user-defined region
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
                    st.image(img_gray, caption="Cropped Image", use_column_width=True)
else:
    st.sidebar.warning("Please upload your Google Cloud Vision API key file.")

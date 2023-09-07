"""Author name: prashant Jha"""

!apt-get install -y poppler-utils
!pip install PyPDF2 pdf2image opencv-python

# Required Libraries
import PyPDF2
import re
import pdf2image
import cv2
import numpy as np
import pandas as pd

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        texts = [pdf_reader.pages[page_num].extract_text() for page_num in range(len(pdf_reader.pages))]
    return texts


# Extract area and region names from the text
def extract_area_data(texts):
    pattern = r"(\w+ AREA):[ \n]+(\d+ sq\.ft\.)"
    area_data = []
    for text in texts:
        matches = re.findall(pattern, text)
        area_dict = {region: area for region, area in matches}
        area_data.append(area_dict)
    return area_data

# Convert PDF to images
def pdf_to_images(pdf_path):
    return pdf2image.convert_from_path(pdf_path)

# Detect north direction using edge-based template matching
def detect_north_direction(image, template):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    img_edges = cv2.Canny(img_gray, 50, 150)
    template_edges = cv2.Canny(template_gray, 50, 150)
    res = cv2.matchTemplate(img_edges, template_edges, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if template.shape[0] > template.shape[1]:
        direction = "North/South"
    else:
        direction = "East/West"
    return direction

# Determine front gate orientation based on north direction
def get_front_gate_orientation(north_direction):
    orientations = {
        "North/South": "South",
        "South/North": "North",
        "East/West": "East",
        "West/East": "West"
    }
    return orientations.get(north_direction, "Unknown")

# Main code
pdf_path = "/content/sample_data/Floorplan/One-Cole-Floorplans.pdf"
template_path = "/content/sample_data/Floorplan/indicator.png"

# Extract data
texts = extract_text_from_pdf(pdf_path)
area_data = extract_area_data(texts)
images = pdf_to_images(pdf_path)
template = cv2.imread(template_path)

# Determine north directions and front gate orientations
north_directions = [detect_north_direction(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), template) for img in images]
front_gate_orientations = [get_front_gate_orientation(direction) for direction in north_directions]

# Create and save DataFrame
df = pd.DataFrame(area_data)
df['North Direction'] = north_directions
df['Front Gate Orientation'] = front_gate_orientations
df['Page Number'] = range(1, len(df) + 1)
df.to_excel("output_data.xlsx", index=False)


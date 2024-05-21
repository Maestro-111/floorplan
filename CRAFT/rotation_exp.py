import cv2
import numpy as np
from matplotlib import pyplot as plt

def detect_rotation_angle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.rad2deg(theta)
            return angle
    return 0  # If no lines detected, assume no rotation

def rotate_image(image, angle):
    # Get image center
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
    return rotated_image

# Load the image

image = cv2.imread('test/pdf_floor_plan_16_8.jpg')

print(image.shape)

#if len(image.shape) > 2:
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect rotation angle
rotation_angle = detect_rotation_angle(image)

# Rotate the image back to its normal orientation
if rotation_angle != 0:
    rotated_image = rotate_image(image, rotation_angle)
    plt.imshow(rotated_image)
    plt.show()
else:
    print("No rotation detected.")

from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage

def apply_sobel_edge_detection(image):
    """Apply Sobel edge detection to an image."""
    # Convert image to grayscale
    gray_image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    # Apply Sobel filter
    dx = ndimage.sobel(gray_image, 0)  # horizontal derivative
    dy = ndimage.sobel(gray_image, 1)  # vertical derivative
    magnitude = np.hypot(dx, dy)  # magnitude
    return magnitude

# Applying Sobel edge detection on the sample image
edges_sample_sobel = apply_sobel_edge_detection(sample_img)

# Visualize the edge detection result
plt.figure(figsize=(6, 6))
plt.imshow(edges_sample_sobel, cmap='gray')
plt.title("Sobel Edge Detection Result")
plt.axis("off")
plt.show()

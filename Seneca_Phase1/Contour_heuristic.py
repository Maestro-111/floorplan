# Apply K-means clustering with 5 clusters
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

def extract_features(contours):
    """Extract area and perimeter features from contours."""
    unique_contours = np.unique(contours)
    features = []

    for contour in unique_contours:
        # Calculate area
        area = np.sum(contours == contour)
        # Calculate perimeter
        perimeter = np.sum(ndimage.binary_erosion(contours == contour) != (contours == contour))
        features.append((contour, area, perimeter))
    
    return features

# Extract features from the detected contours
contour_features = extract_features(contours_sample)

# Displaying the first few features for inspection
contour_features[:5]


kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(X)

# Mapping each contour to its cluster label
contour_cluster_map = dict(zip([c[0] for c in contour_features], clusters))

# Replace each contour label in the image with its cluster label
clustered_image = np.copy(contours_sample)
for contour, cluster in contour_cluster_map.items():
    clustered_image[contours_sample == contour] = cluster

# Visualize the floor plan with heuristic labels
plt.figure(figsize=(8, 8))
plt.imshow(clustered_image, cmap='tab10')

# Add labels
for cluster, label in cluster_labels.items():
    plt.text(10, 10 + cluster * 20, f"Cluster {cluster}: {label}", 
             color='white', 
             bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.5'))

plt.title("Heuristic Labeled Floor Plan")
plt.axis("off")
plt.show()

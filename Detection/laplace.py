import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_laplacian_variance(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

blurry_images_dir = 'blur'
non_blurry_images_dir = 'not_blur'

blurry_variances = []
non_blurry_variances = []

for img_name in os.listdir(blurry_images_dir):
    img_path = os.path.join(blurry_images_dir, img_name)
    variance = compute_laplacian_variance(img_path)
    blurry_variances.append(variance)

for img_name in os.listdir(non_blurry_images_dir):
    img_path = os.path.join(non_blurry_images_dir, img_name)
    variance = compute_laplacian_variance(img_path)
    non_blurry_variances.append(variance)

plt.figure(figsize=(12, 6))
plt.hist(blurry_variances, bins=50, alpha=0.5, label='Blurry Images')
plt.hist(non_blurry_variances, bins=50, alpha=0.5, label='Non-Blurry Images')
plt.title('Distribution of Laplacian Variance for Blurry and Non-Blurry Images')
plt.xlabel('Laplacian Variance')
plt.ylabel('Frequency')
plt.legend()
plt.show()

all_variances = blurry_variances + non_blurry_variances

max_acc = 0
percentile = 0
best_threshold = 0

for i in range(1, 100):
    threshold = np.percentile(all_variances, i)
    accuracy = (sum(var < threshold for var in blurry_variances) + sum(var >= threshold for var in non_blurry_variances)) / len(all_variances)
    if accuracy > max_acc:
        max_acc = accuracy
        percentile = i
        best_threshold = threshold

print(f"Best threshold: {best_threshold}")
print(f"Best percentile: {percentile}")
print(f"Best accuracy: {max_acc}")
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_blur_fft(path, size=60, vis=True):
    image = cv2.imread(path)
    image = imutils.resize(image, width=500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    (h,w) = image.shape
    (cX, cY) = (int(w/2.0), int(h/2.0))
    fft = np.fft.fft2(image)
    fftShift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fftShift))

    fftShift[cY - size:cY + size, cX - size:cX + size] = 0
    fftShift = np.fft.ifftshift(fftShift)
    recon = np.fft.ifft2(fftShift)

    magnitude = 20 * np.log(np.abs(recon))
    mean = np.mean(magnitude)
    return mean

blurry_images_dir = 'blur'
non_blurry_images_dir = 'not_blur'

blurry_variances = []
non_blurry_variances = []

for img_name in os.listdir(blurry_images_dir):
    img_path = os.path.join(blurry_images_dir, img_name)
    mean = detect_blur_fft(img_path, size=60)
    blurry_variances.append(mean)

for img_name in os.listdir(non_blurry_images_dir):
    img_path = os.path.join(non_blurry_images_dir, img_name)
    mean = detect_blur_fft(img_path, size=60)
    non_blurry_variances.append(mean)

plt.figure(figsize=(12, 6))
plt.hist(blurry_variances, bins=50, alpha=0.5, label='Blurry Images')
plt.hist(non_blurry_variances, bins=50, alpha=0.5, label='Non-Blurry Images')
plt.title('Distribution of Mean Magnitude for Blurry and Non-Blurry Images')
plt.xlabel('Mean Magnitude')
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
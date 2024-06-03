import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def extract_features(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    return [laplacian_var, mean_intensity, std_intensity]

def prepare_dataset(image_dir, label):
    features = []
    labels = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        feature_vector = extract_features(img_path)
        features.append(feature_vector)
        labels.append(label)
    return features, labels

blurry_features, blurry_labels = prepare_dataset('blur', 0)
non_blurry_features, non_blurry_labels = prepare_dataset('not_blur', 1)

features = np.array(blurry_features + non_blurry_features)
labels = np.array(blurry_labels + non_blurry_labels)

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")

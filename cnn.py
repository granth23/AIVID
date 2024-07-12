import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

IMG_SIZE = 128

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    return image

def prepare_image_dataset(image_dir, label):
    images = []
    labels = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        image = load_and_preprocess_image(img_path)
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

blurry_images, blurry_labels = prepare_image_dataset('old/plate/blur', 0)
non_blurry_images, non_blurry_labels = prepare_image_dataset('old/plate/not_blur', 1)

images = np.concatenate([blurry_images, non_blurry_images], axis=0)
labels = np.concatenate([blurry_labels, non_blurry_labels], axis=0)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save('cnn_model.h5')

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")

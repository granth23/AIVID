from flask import Flask, render_template, request
import albumentations as A
import cv2

app = Flask(__name__)

# Dictionary mapping augmentation types to Albumentations functions
augmentation_functions = {
    'gaussian_blur': A.GaussianBlur,
    'motion_blur': A.MotionBlur,
    'median_blur': A.MedianBlur,
    'gaussian_noise': A.GaussNoise,
    'brightness': A.RandomBrightnessContrast,
    'contrast': A.RandomBrightnessContrast,
    'hue': A.HueSaturationValue,
    'saturation': A.HueSaturationValue
}

# Dictionary mapping augmentation types to display labels
augmentation_labels = {
    'gaussian_blur': {'label': 'Gaussian Blur', 'min': 1, 'max': 15, 'step': 2, 'value': 3},
    'motion_blur': {'label': 'Motion Blur', 'min': 1, 'max': 15, 'step': 2, 'value': 3},
    'median_blur': {'label': 'Median Blur', 'min': 1, 'max': 15, 'step': 2, 'value': 3},
    'gaussian_noise': {'label': 'Gaussian Noise', 'min': 1, 'max': 15, 'step': 2, 'value': 3},
    'brightness': {'label': 'Brightness', 'min': -1, 'max': 1, 'step': 0.1, 'value': 0},
    'contrast': {'label': 'Contrast', 'min': -1, 'max': 1, 'step': 0.1, 'value': 0},
    'hue': {'label': 'Hue', 'min': -1, 'max': 1, 'step': 0.1, 'value': 0},
    'saturation': {'label': 'Saturation', 'min': -1, 'max': 1, 'step': 0.1, 'value': 0}
}

@app.route('/')
def index():
    return render_template('index.html', augmentation_labels=augmentation_labels)

@app.route('/<int:image_id>', methods=['POST'])
def augment_image_endpoint(image_id):
    if request.method == 'POST':
        limit = float(request.form['limit'])
        limit_type = request.form['limit_type']
        image_path = f'static/samples/original.jpg'
        augmented_image_path = f'static/samples/result{image_id}.jpg'
        image = cv2.imread(image_path)

        # Apply the selected augmentation
        augmented_image = apply_augmentation(image, limit, limit_type)

        # Save augmented image
        cv2.imwrite(augmented_image_path, augmented_image)

        return render_template('index.html', image_id=image_id, augmentation_labels=augmentation_labels)

def convert_range(x, a, b, c, d):
    x_normalized = (x - a) / (b - a)
    y = c + (d - c) * x_normalized
    return y

def apply_augmentation(image, limit, limit_type):
    if limit_type in augmentation_functions:
        aug_function = augmentation_functions[limit_type]
        if limit_type in ['brightness', 'contrast']:
            if limit_type == 'brightness':
                limit = convert_range(limit, -1, 1, -0.5, 0.5)
                return aug_function(brightness_limit=limit, contrast_limit=0, p=1)(image=image)['image']
            else:
                limit = convert_range(limit, -1, 1, -0.5, 0.5)
                return aug_function(brightness_limit=0, contrast_limit=limit, p=1)(image=image)['image']
        elif limit_type in ['hue', 'saturation']:
            if limit_type == 'hue':
                limit = convert_range(limit, -1, 1, -20, 20)
                return aug_function(hue_shift_limit=limit, sat_shift_limit=0, p=1)(image=image)['image']
            else:
                limit = convert_range(limit, -1, 1, -100, 100)
                return aug_function(hue_shift_limit=0, sat_shift_limit=limit, p=1)(image=image)['image']
        else:
            if limit_type == 'gaussian_noise':
                limit = convert_range(limit, 1, 15, 100, 1000)
                return aug_function(var_limit=limit, p=1)(image=image)['image']
            return aug_function(blur_limit=limit, p=1)(image=image)['image']
    else:
        return image

if __name__ == '__main__':
    app.run(debug=True)

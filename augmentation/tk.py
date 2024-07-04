import albumentations as A
import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def augment_image(image, limits):
    transform = A.Compose([
        A.GaussianBlur(blur_limit=(limits['gaussian_blur']['min'], limits['gaussian_blur']['max']), p=0.5),
        A.MotionBlur(blur_limit=(limits['motion_blur']['min'], limits['motion_blur']['max']), p=0.5),
        A.MedianBlur(blur_limit=(limits['median_blur']['min'], limits['median_blur']['max']), p=0.5),
        A.GaussNoise(var_limit=(limits['gaussian_noise']['min'], limits['gaussian_noise']['max']), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(limits['brightness_contrast']['brightness_min'], limits['brightness_contrast']['brightness_max']),
                                   contrast_limit=(limits['brightness_contrast']['contrast_min'], limits['brightness_contrast']['contrast_max']), p=0.5),
        A.HueSaturationValue(hue_shift_limit=(limits['hue_saturation']['hue_min'], limits['hue_saturation']['hue_max']),
                             sat_shift_limit=(limits['hue_saturation']['saturation_min'], limits['hue_saturation']['saturation_max']), p=0.5)
    ])

    return transform(image=image)['image']

def augment_images(input_dir, output_dir, limits):
    for image_path in os.listdir(input_dir):
        image = cv2.imread(f'{input_dir}/{image_path}')
        for i in range(5):
            augmented_image = augment_image(image, limits)
            new_image_path = f'{output_dir}/{image_path.split(".")[0]}_aug_{i}.jpg'
            cv2.imwrite(new_image_path, augmented_image)

def select_directory(entry):
    directory = filedialog.askdirectory()
    entry.delete(0, tk.END)
    entry.insert(0, directory)

def start_augmentation():
    input_dir = input_dir_entry.get()
    output_dir = output_dir_entry.get()

    if not input_dir or not os.path.exists(input_dir):
        messagebox.showerror("Error", "Invalid input directory selected!")
        return

    if not output_dir or not os.path.exists(output_dir):
        messagebox.showerror("Error", "Invalid output directory selected!")
        return

    limits = {
        'gaussian_blur': {'min': blur1_min.get(), 'max': blur1_max.get()},
        'motion_blur': {'min': blur2_min.get(), 'max': blur2_max.get()},
        'median_blur': {'min': blur3_min.get(), 'max': blur3_max.get()},
        'gaussian_noise': {'min': noise_min.get(), 'max': noise_max.get()},
        'brightness_contrast': {
            'brightness_min': bright_min.get(), 'brightness_max': bright_max.get(),
            'contrast_min': contrast_min.get(), 'contrast_max': contrast_max.get()
        },
        'hue_saturation': {
            'hue_min': hue_min.get(), 'hue_max': hue_max.get(),
            'saturation_min': sat_min.get(), 'saturation_max': sat_max.get()
        }
    }

    augment_images(input_dir, output_dir, limits)
    messagebox.showinfo("Success", "Augmentation complete! Check the output directory for the augmented images.")

def set_to_nearest_odd(scale):
    value = scale.get()
    if value % 2 == 0:
        scale.set(value + 1 if value < scale.cget('to') else value - 1)

# Setup the Tkinter window
root = tk.Tk()
root.title("Image Augmentation")
root.geometry("600x700")

tk.Label(root, text="Input Directory:").grid(row=0, column=0, sticky=tk.E)
input_dir_entry = tk.Entry(root, width=50)
input_dir_entry.grid(row=0, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=lambda: select_directory(input_dir_entry)).grid(row=0, column=2)

tk.Label(root, text="Output Directory:").grid(row=1, column=0, sticky=tk.E)
output_dir_entry = tk.Entry(root, width=50)
output_dir_entry.grid(row=1, column=1, padx=10, pady=5)
tk.Button(root, text="Browse", command=lambda: select_directory(output_dir_entry)).grid(row=1, column=2)

tk.Label(root, text="Gaussian Blur").grid(row=2, column=0, columnspan=3)
blur1_min = tk.Scale(root, from_=1, to_=15, orient=tk.HORIZONTAL, label="Min", tickinterval=2, command=lambda _: set_to_nearest_odd(blur1_min))
blur1_min.grid(row=3, column=0)
blur1_max = tk.Scale(root, from_=1, to_=15, orient=tk.HORIZONTAL, label="Max", tickinterval=2, command=lambda _: set_to_nearest_odd(blur1_max))
blur1_max.grid(row=3, column=1)

tk.Label(root, text="Motion Blur").grid(row=4, column=0, columnspan=3)
blur2_min = tk.Scale(root, from_=1, to_=15, orient=tk.HORIZONTAL, label="Min", tickinterval=2, command=lambda _: set_to_nearest_odd(blur2_min))
blur2_min.grid(row=5, column=0)
blur2_max = tk.Scale(root, from_=1, to_=15, orient=tk.HORIZONTAL, label="Max", tickinterval=2, command=lambda _: set_to_nearest_odd(blur2_max))
blur2_max.grid(row=5, column=1)

tk.Label(root, text="Median Blur").grid(row=6, column=0, columnspan=3)
blur3_min = tk.Scale(root, from_=1, to_=15, orient=tk.HORIZONTAL, label="Min", tickinterval=2, command=lambda _: set_to_nearest_odd(blur3_min))
blur3_min.grid(row=7, column=0)
blur3_max = tk.Scale(root, from_=1, to_=15, orient=tk.HORIZONTAL, label="Max", tickinterval=2, command=lambda _: set_to_nearest_odd(blur3_max))
blur3_max.grid(row=7, column=1)

tk.Label(root, text="Gaussian Noise").grid(row=8, column=0, columnspan=3)
noise_min = tk.Scale(root, from_=0, to_=100, orient=tk.HORIZONTAL, label="Min")
noise_min.grid(row=9, column=0)
noise_max = tk.Scale(root, from_=0, to_=100, orient=tk.HORIZONTAL, label="Max")
noise_max.grid(row=9, column=1)

tk.Label(root, text="Brightness").grid(row=10, column=0, columnspan=3)
bright_min = tk.Scale(root, from_=-1.0, to_=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Min")
bright_min.grid(row=11, column=0)
bright_max = tk.Scale(root, from_=-1.0, to_=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Max")
bright_max.grid(row=11, column=1)

tk.Label(root, text="Contrast").grid(row=12, column=0, columnspan=3)
contrast_min = tk.Scale(root, from_=-1.0, to_=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Min")
contrast_min.grid(row=13, column=0)
contrast_max = tk.Scale(root, from_=-1.0, to_=1.0, resolution=0.1, orient=tk.HORIZONTAL, label="Max")
contrast_max.grid(row=13, column=1)

tk.Label(root, text="Hue").grid(row=14, column=0, columnspan=3)
hue_min = tk.Scale(root, from_=-180, to_=180, orient=tk.HORIZONTAL, label="Min")
hue_min.grid(row=15, column=0)
hue_max = tk.Scale(root, from_=-180, to_=180, orient=tk.HORIZONTAL, label="Max")
hue_max.grid(row=15, column=1)

tk.Label(root, text="Saturation").grid(row=16, column=0, columnspan=3)
sat_min = tk.Scale(root, from_=-100, to_=100, orient=tk.HORIZONTAL, label="Min")
sat_min.grid(row=17, column=0)
sat_max = tk.Scale(root, from_=-100, to_=100, orient=tk.HORIZONTAL, label="Max")
sat_max.grid(row=17, column=1)

tk.Button(root, text="Start Augmentation", command=start_augmentation).grid(row=18, column=0, columnspan=3, pady=20)

root.mainloop()

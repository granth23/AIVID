import os
import shutil

def create_image_copies(file, destination_folder):
    for i in range(1, 17):
        shutil.copy(file, f'{destination_folder}/result{i}.jpg')

file = 'C:/Users/Granth Bagadia/Code/Projects/Dev/AIVID/augmentation/static/samples/original.jpg'
destination_folder = 'C:/Users/Granth Bagadia/Code/Projects/Dev/AIVID/augmentation/static/samples/'

create_image_copies(file, destination_folder)
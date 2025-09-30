# Author : Vriti Sharma

# Importing Libraries
import numpy as np
import pandas as pd
import os
import cv2

from pdf2image import convert_from_path
from PIL import Image 
import PIL 
import matplotlib.pyplot as plt

# Changing UNCLUTTER from 200 to 600 and 300 DPI
# SOURCE PATH
parent_dir = r'../200_DPI/Un_cluttered'

# looping through folders
folder_list = os.listdir(parent_dir)

# DESTINATION PATH
parent_dir_200_ = r'../200_DPI/Un_cluttered_images'
parent_dir_300 = r'../300_DPI/Un_cluttered'
parent_dir_600 = r'../600_DPI/Un_cluttered'

os.makedirs(parent_dir_200_, exist_ok=True)
os.makedirs(parent_dir_300, exist_ok=True)
os.makedirs(parent_dir_600, exist_ok=True)

for folder in folder_list :
    folder_name = folder[:-4]

    pdf_path = os.path.join(parent_dir, folder)

    img = convert_from_path(pdf_path, 200)
    img_ = convert_from_path(pdf_path, 300)
    img__ = convert_from_path(pdf_path, 600)

    img = np.array(img).squeeze()
    img_ = np.array(img_).squeeze()
    img__ = np.array(img__).squeeze()

    print(img.shape)
    print(img_.shape)
    print(img__.shape)

    img = Image.fromarray(img, 'RGB')
    img_ = Image.fromarray(img_, 'RGB')
    img__ = Image.fromarray(img__, 'RGB')

    img.save(os.path.join(parent_dir_200_, f'{folder_name}.png'))    
    img_.save(os.path.join(parent_dir_300, f'{folder_name}.png')) 
    img__.save(os.path.join(parent_dir_600, f'{folder_name}.png')) 

# Changing CLUTTER from 600 to 300 DPI
# SOURCE PATH
parent_dir = r'../600_DPI/Clutter'

# looping through folders
folder_list = os.listdir(parent_dir)

# DESTINATION PATH
parent_dir_300 = r'../300_DPI/Clutter'
os.makedirs(parent_dir_300, exist_ok=True)

new_dpi=(300, 300)

for file in folder_list :
    img_path = os.path.join(parent_dir, file)
    img = Image.open(img_path)

    current_size = img.size
    current_dpi = img.info.get('dpi', (72, 72))

    scale_factor = new_dpi[0] / current_dpi[0]
    new_size = (int(current_size[0] * scale_factor), int(current_size[1] * scale_factor))

    img = img.resize(new_size, Image.Resampling.LANCZOS)

    current_dpi = img.info.get('dpi', (72, 72))
    img.save(os.path.join(parent_dir_300, file), dpi=new_dpi) 
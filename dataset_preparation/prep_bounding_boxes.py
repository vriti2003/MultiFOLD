# Author : Vriti Sharma

# Importing Libraries
import pytesseract
import cv2
from pdf2image import convert_from_path
from pytesseract import Output
import numpy as np
import pandas as pd
import os
import json

from PyPDF2 import PdfReader
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt

# Converting pdf to images
parent_dir = '../600_DPI'
un_cluttered_folder_path = '../600_DPI/Un_cluttered'
cluttered_folder_path = '../600_DPI/Cluttered'

# Utilites
preprocessed_store_loc = 'sentence_images_600_600'
bounding_boxes = 'bounding_boxes_un_cluttered'

def store_bounding_box(un_c_img, c_img_, file_name):
    un_c_img_ = cv2.cvtColor(np.array(un_c_img), cv2.COLOR_RGB2GRAY)
    _, un_c_img_ = cv2.threshold(un_c_img_, 127, 255, cv2.THRESH_BINARY)

    data = pytesseract.image_to_data(un_c_img_, config='--oem 3 --psm 6', output_type='dict')

    df = pd.DataFrame(data)
    df = df[df["conf"] > 0]

    # For a paragraph and line combination, create line text and bounding box dimension
    page_par_line_dict = {}
    for index, row in df.iterrows():
        page_par_line = f"{row['par_num']}_{row['line_num']}"
        if(page_par_line not in page_par_line_dict):
            page_par_line_dict[page_par_line] = {"uc_text": str(row["text"]) + " ", "box": (row['left'], row['top'], row['left'] + row['width'], row['top'] + row['height'])}
        else:
            page_par_line_dict[page_par_line]["uc_text"] = page_par_line_dict[page_par_line]["uc_text"] + str(row["text"]) + " "
            page_par_line_dict[page_par_line]['box'] = (min(page_par_line_dict[page_par_line]['box'][0], row['left']), 
                                                  min(page_par_line_dict[page_par_line]['box'][1], row['top']), 
                                                  max(page_par_line_dict[page_par_line]['box'][2], row['left'] + row['width']), 
                                                  max(page_par_line_dict[page_par_line]['box'][3], row['top'] + row['height']))
    
    # COMMENT 1 : Store the CSV file to store the predicted Un-cluttered text
    i = 1
    idx = []
    sentence = []
    page = []
    
    for entry in page_par_line_dict:
        entry_value = page_par_line_dict[entry]
        idx.append(i)
        sentence.append(entry_value['uc_text'])
        page.append(file_name)
        i += 1
        
    data = {
        'page' : page,
        'index' : idx,
        'sentence' : sentence
    }

    df = pd.DataFrame(data)
    df.to_csv("OCR_uncluttered_data.csv", mode='a', header=False, index=False)

    # COMMENT 2 : Storing the sentence level images of Cluttered and Un-Cluttered 

    os.makedirs(os.path.join(parent_dir, bounding_boxes, file_name), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, preprocessed_store_loc, 'Un_cluttered', file_name), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, preprocessed_store_loc, 'Cluttered', file_name), exist_ok=True)

    os.makedirs(os.path.join('un_cluttered', file_name), exist_ok=True)
    os.makedirs(os.path.join('cluttered', file_name), exist_ok=True)

    with open(os.path.join(parent_dir, bounding_boxes, file_name, f'{file_name}.json'), "w") as outfile: 
        json.dump(page_par_line_dict, outfile)

    for i, line in enumerate(page_par_line_dict.values()):
        if line['box'] is not None:
            X, Y, X_, Y_ = line['box'][0], line['box'][1], line['box'][2], line['box'][3]

            un_c_img_cropped_image = un_c_img[Y:Y_, X:X_]

            c_img_cropped_image = c_img_[Y:Y_, X:X_]

            cv2.imwrite(os.path.join(parent_dir, preprocessed_store_loc, 'Un_cluttered', file_name, f'{i}.png'), un_c_img_cropped_image)
            cv2.imwrite(os.path.join(parent_dir, preprocessed_store_loc, 'Cluttered', file_name, f'{i}.png'), c_img_cropped_image)

# Performing OCR and drawing Bounding Boxes for each uncuttered page and applied on cluttered page
onlyfiles = [f for f in listdir(un_cluttered_folder_path) if isfile(join(un_cluttered_folder_path, f))]

for file in onlyfiles:
    file_name = file[:-4]

    un_cluttered_file_path = os.path.join(un_cluttered_folder_path, file)
    cluttered_file_path = os.path.join(cluttered_folder_path, file)

    un_cluttered_file_path = os.path.join(un_cluttered_folder_path, f'{file_name}.png')
    cluttered_file_path = os.path.join(cluttered_folder_path, f'{file_name}.png')
    
    un_c_img = convert_from_path(un_cluttered_file_path, 200)
    un_c_img_ = np.array(un_c_img).squeeze()

    c_img = convert_from_path(cluttered_file_path, 200)
    c_img_ = np.array(c_img).squeeze()
    
    un_c_img_ = cv2.imread(un_cluttered_file_path)
    c_img_ = cv2.imread(cluttered_file_path)

    store_bounding_box(un_c_img_, c_img_, file_name)
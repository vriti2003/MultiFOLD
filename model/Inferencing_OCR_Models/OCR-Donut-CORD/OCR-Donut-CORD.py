# ### Importing libraries
import os
from PIL import Image
from transformers import DonutProcessor, pipeline
import pandas as pd

# ### Loading model and Tokenizer
# Load the processor
processor = DonutProcessor.from_pretrained("jinhybr/OCR-Donut-CORD")
ocr_pipeline = pipeline("image-to-text", model="jinhybr/OCR-Donut-CORD")

# ### Utilities
# Function for OCR
def ocr_image(src_img):
    # Get the prediction from the OCR model
    output_text = ocr_pipeline(src_img)[0]["generated_text"]
    return output_text

# ### Loading the clutter images
parent_dir = '../sentence_images_/cluttered_'

# looping through folders
folder_list = os.listdir(parent_dir)

# ### Inferencing the model
page, idx, sentence = [], [], []

for folder in folder_list :
    files = os.listdir(os.path.join(parent_dir, folder))

    img_files = [f for f in files if os.path.isfile(os.path.join(parent_dir, folder, f))]
    for img in files:
        page.append(folder)
        idx.append(img[:-4])

        full_image = Image.open(os.path.join(parent_dir, folder, img)).convert("RGB")
        text = ocr_image(full_image)
        sentence.append(text)

data = {
        'page' : page,
        'index' : idx,
        'sentence' : sentence
    }

df = pd.DataFrame(data)
df.to_csv("ocr_donut_cord_ocr.csv", mode='a', header=False, index=False)




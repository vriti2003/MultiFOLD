import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from jiwer import cer
from PIL import Image, ImageOps
from IPython.display import display
import pandas as pd

# Load the model and processor
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-printed')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-printed')
# processor = TrOCRProcessor.from_pretrained('../FinetuningTrOCR/version/final_9')
model = VisionEncoderDecoderModel.from_pretrained('../FinetuningTrOCR/version/final_9')
# ### Utilities
# Function for OCR
def ocr_image(src_img):
    pixel_values = processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ### Load the cluttered images

parent_dir = '../sentence_images_200_600_/cluttered_'

# looping through folders
folder_list = os.listdir(parent_dir)

# ### Inferencing on the model

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
df.to_csv("tr_ocr.csv", mode='a', header=False, index=False)



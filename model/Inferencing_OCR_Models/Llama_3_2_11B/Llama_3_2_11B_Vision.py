# ### Importing Libraries
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
import pandas as pd

# ### Hugging Face login
# from huggingface_hub import login
# login()

# ### Loading the Model and Tokenizer
# Initialize processor and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("meta-llama/Llama-3.2-11B-Vision")
model = AutoModelForImageTextToText.from_pretrained(
    "meta-llama/Llama-3.2-11B-Vision",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)

# ### Utilities
# Function for OCR
def ocr_image(src_img):
    # Get the prediction from the OCR model
    prompt = "<|image|>Perform OCR on the image and extract the text string"
    inputs = processor(src_img, prompt, return_tensors="pt").to(device)

    # Generate output
    output_ids = model.generate(**inputs,max_new_tokens=50)
    output_text = processor.decode(output_ids[0], skip_special_tokens=True)
    
    return output_text.strip()

# ### Loading the cluttering images
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
df.to_csv("Llama_3_2_11B_vision_ocr.csv", mode='a', header=False, index=False)



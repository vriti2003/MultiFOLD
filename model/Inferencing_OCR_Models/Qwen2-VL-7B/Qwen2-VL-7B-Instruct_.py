
import os
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from jiwer import cer
from PIL import Image, ImageOps
from IPython.display import display
import pandas as pd

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

# Function for OCR
def ocr_image(src_img):

    messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": src_img,
            },
            {"type": "text", "text": "Perform OCR on the image."},
        ],
    }]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text

parent_dir = '../sentence_images_/cluttered_'

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
df.to_csv("qwen2_vl-7b_i_ocr.csv", mode='a', header=False, index=False)
# Author : Vriti Sharma

import pytesseract
import cv2
import numpy as np
import pandas as pd
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image

from PyPDF2 import PdfReader
from os import listdir
from os.path import isfile, join
from sklearn.model_selection import train_test_split
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, TrOCRProcessor, VisionEncoderDecoderModel, default_data_collator

from evaluate import load
import wandb
from datetime import datetime

cer_metric = load("cer")
wer_metric = load("wer")

now = datetime.now()
os.environ["WANDB_WATCH"]="false"
wandb.init(project="tr_OCR_fine_tune", name=f'TR_OCR_{str(now)}')

# Finetuning of TR_OCR 
train_df = pd.read_csv('dataset/train.csv')
val_df = pd.read_csv('dataset/val.csv')
test_df = pd.read_csv('dataset/test.csv')

model_name = "microsoft/trocr-base-printed"
processor = TrOCRProcessor.from_pretrained(model_name)

class MultiFOLD(Dataset):
    def __init__(self, df, processor, max_target_length=128):
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['filename'][idx]
        text = self.df['s'][idx]
        # prepare image (i.e. resize + normalize)
        image = Image.open(file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

train_dataset = MultiFOLD(df=train_df, processor=processor)
eval_dataset = MultiFOLD(df=val_df, processor=processor)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

class CustomVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    def forward(self, pixel_values=None, labels=None, **kwargs):
        # Ignore the extra argument
        kwargs.pop("num_items_in_batch", None)
        return super().forward(pixel_values=pixel_values, labels=labels, **kwargs)

model = CustomVisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 128
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer_ = cer_metric.compute(predictions=pred_str, references=label_str)
    wer_ = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer_, "wer": wer_}

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    eval_strategy="steps",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=500,
    fp16=True, 
    output_dir="./chkpt",
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
    report_to='wandb'
)

# Instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

# Train the model
trainer.train()

wandb.finish()
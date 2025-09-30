# Author : Vriti Sharma

"""### Installing libraries"""

# ! pip install -U datasets transformers
# ! pip install -U evaluate sacrebleu
"""### Importing libraries"""

import os
import pandas as pd
import numpy as np
import logging

from transformers import AutoModelForSeq2SeqLM, Trainer, AutoTokenizer, DataCollatorForSeq2Seq, TrainerCallback, EarlyStoppingCallback
from datasets import Dataset
from evaluate import load
import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

parent_folder = 'folder'
os.makedirs(parent_folder)

#setting the logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename=os.path.join(parent_folder, 'debug_byt5_large_e_10_bs_8.log'), filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

# for wandb intialisation
os.environ["WANDB_PROJECT"]="16_oct_24_byt5_large_e_10_bs_8" # name of W&B project

# Setting threshold for printing on the console screen 
np.set_printoptions(threshold = np.inf)

"""### Loading the byt5-large model and corresponding tokeniser"""

# loading the metrics
wer_metric = load("wer")
cer_metric = load("cer")

# Constant
MAX_LENGTH = 60

model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-large")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-large")
data_collator = DataCollatorForSeq2Seq(tokenizer)

"""### Loading data"""

df_train = pd.read_excel('../Data/hindi_train.xlsx')
df_val = pd.read_excel('../Data/hindi_train.xlsx')
df_test = pd.read_excel('../Data/hindi_train.xlsx')

logger.debug("printing the columns of train : %s", df_train.columns)

ocr_train = np.asarray(df_train['ocr_output'])
gt_train = np.asarray(df_train['correct_text'])

logger.debug("printing the columns of val : %s", df_val.columns)

ocr_val = np.asarray(df_val['ocr_output'])
gt_val = np.asarray(df_val['correct_text'])

logger.debug("printing the columns of test : %s", df_test.columns)

ocr_test = np.asarray(df_test['ocr_output'])
gt_test = np.asarray(df_test['correct_text'])

# preparing the dictionaries
train = {
    'input' : ocr_train,
    'target' : gt_train
}

val = {
    'input' : ocr_val,
    'target' : gt_val
}

test = {
	'input' : ocr_test,
	'target' : gt_test
}

ds_train = Dataset.from_dict(train)
ds_eval = Dataset.from_dict(val)
ds_test = Dataset.from_dict(test)

ds_train = ds_train.map(lambda x: tokenizer(x["input"], truncation=True, padding="max_length"))

ds_train = ds_train.map(lambda y: {"labels": tokenizer(y["target"], truncation=True, padding="max_length")['input_ids']})

ds_eval = ds_eval.map(lambda x: tokenizer(x["input"], truncation=True, padding="max_length"))

ds_eval = ds_eval.map(lambda y: {"labels": tokenizer(y["target"], truncation=True, padding="max_length")['input_ids']})

# compute metrics function
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    logger.debug("=============== VALIDATION DATA STATISTICS ===============")
    logger.debug('data of preds under compute metrics : %s', preds)
    logger.debug('data of labels under compute metrics : %s', labels)

    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, clean_up_tokenization_spaces = True, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, clean_up_tokenization_spaces = True, skip_special_tokens=True)

    # Some simple post-processing
    # decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    logger.debug('type of preds under compute metrics : %s', type(decoded_preds))
    logger.debug('type of labels under compute metrics : %s', type(decoded_labels))
    logger.debug('len of preds under compute metrics : %s', len(decoded_preds))
    logger.debug('len of labels under compute metrics : %s', len(decoded_labels))
    logger.debug('data of decoded_preds under compute metrics : %s', decoded_preds)
    logger.debug('data of decoded_labels under compute metrics : %s', decoded_labels)

    # Calculate CER
    cer_results = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    logger.debug("CER under compute metrics : %s", cer_results)

	# Calculate WER
    wer_results = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)
    logger.debug("WER under compute metrics : %s", wer_results)

    # storing the data in csv file 
    store_results = {
        'decoded_preds' : decoded_preds,
        'decoded_labels' : decoded_labels
    }

    cer_wer_results = {
        'cer' : cer_results,
        'wer' : wer_results
    }

    logger.debug("Predictions : %s", store_results)
    df = pd.DataFrame(store_results)
    df.to_csv(os.path.join(parent_folder, "16_oct_24_byt5_base_e_10_bs_8_result.csv"))

    return cer_wer_results
    
# defining the Callback
class MetricCallback(TrainerCallback):
    def __init__(self, tokenizer, train_dataset, eval_dataset, trainer):
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self._trainer = trainer
        self.train_loss = []
        self.train_cer = []
        self.train_wer = []
        self.val_loss = []
        self.val_cer = []
        self.val_wer = []

    def on_epoch_end(self, args, state, control, **kwargs):
        train_output = self._trainer.predict(test_dataset=self.train_dataset)
        logger.debug("train output in metrics callback", train_output)
        self.train_loss.append(train_output[2]['test_loss'])
        self.train_cer.append(train_output[2]['test_cer'])
        self.train_wer.append(train_output[2]['test_wer'])

        val_output = self._trainer.predict(test_dataset=self.eval_dataset)
        logger.debug("val output in metrics callback", val_output)
        self.val_loss.append(val_output[2]['test_loss'])
        self.val_cer.append(val_output[2]['test_cer'])
        self.val_wer.append(val_output[2]['test_wer'])

    def on_train_end(self, args, state, control, **kwargs):
        # save the stats to csv file
        results = {
            'train_loss' : self.train_loss,
            'train_cer' : self.train_cer,
            'train_wer' : self.train_wer,
            'val_loss' : self.val_loss,
            'val_cer' : self.val_cer,
            'val_wer' : self.val_wer,
        }

        results_ = pd.DataFrame(results)
        results_.to_csv(os.path.join(parent_folder, "results.csv"))

# Early Stopping Callback
early_stop = EarlyStoppingCallback(5)

# Loading Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=os.path.join(parent_folder, "16_oct_24_byt5_large_e_10_bs_8"),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    seed=42,
    data_seed=42,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    do_train=True,
    do_eval=True,
    disable_tqdm=False,
    predict_with_generate=True,
    num_train_epochs=10,
    # logging_steps=1000,  # set to 1000 for full training : 500
    save_steps=500,    # set to 500 for full training : 500
    eval_steps=8000,     # set to 8000 for full training : None
    warmup_steps=2000,   # set to 2000 for full training : 0
    # max_steps=1,     # delete for full training
    overwrite_output_dir=True, 
    metric_for_best_model='eval_loss',
    load_best_model_at_end = True,
    report_to="wandb",  # enable logging to W&B
    run_name="16_oct_24_byt5_large_e_10_bs_8",  # name of the W&B run
    logging_steps=1,
    no_cuda=False
)


# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_eval,
    data_collator=data_collator,
    compute_metrics = compute_metrics,
    callbacks=[early_stop]
)

trainer.add_callback(MetricCallback(tokenizer, ds_train, ds_eval, trainer)) 

## Training
history = trainer.train()
logger.debug('Train history : %s', history)

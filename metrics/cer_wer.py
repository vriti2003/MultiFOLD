# Author : Vriti Sharma

# Importing libraries
import numpy as np
import pandas as pd
import csv

# Reading the csv files
ground_truth = pd.read_excel('../sorted_files/OCR_clean_data.xlsx')
tr_ocr = pd.read_excel('../sorted_files/tr_ocr.xlsx')
qwen_ocr = pd.read_excel('../sorted_files/qwen2_vl-7b_i_ocr_refined.xlsx')
llama_ocr = pd.read_excel('../sorted_files/Llama_3_2_11B_vision_ocr_refined.xlsx')
donut_ocr = pd.read_excel('../sorted_files/ocr_donut_cord_ocr_refined.xlsx')
deepseek_ocr = pd.read_excel('../sorted_files/deepseek_vl2_refined.xlsx')

ground_truth_preds = ground_truth.iloc[:, 2]
qwen_ocr_preds = qwen_ocr.iloc[:, 4]
llama_ocr_preds = llama_ocr.iloc[:, 4]
donut_ocr_preds = donut_ocr.iloc[:, 4]
tr_ocr_preds = tr_ocr.iloc[:, 2]
deepseek_preds = deepseek_ocr.iloc[:, 4]

# CER and WER functions
# Function to calculate Word Error Rate (WER)
def wer(reference, hypothesis):
    ref = reference.split()
    hyp = hypothesis.split()
    
    # Creating a matrix to calculate the minimum edit distance
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[i][j] = j  # Insertions
            elif j == 0:
                d[i][j] = i  # Deletions
            else:
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1,   # Deletion
                              d[i][j-1] + 1,   # Insertion
                              d[i-1][j-1] + cost)  # Substitution

    # WER is the number of edits divided by the number of words in the reference
    return d[len(ref)][len(hyp)] / len(ref)

# Function to calculate Character Error Rate (CER)
def cer(reference, hypothesis):
    ref = reference.replace(" ", "")  # Remove spaces for character comparison
    hyp = hypothesis.replace(" ", "")  # Remove spaces for character comparison
    
    # Creating a matrix to calculate the minimum edit distance
    d = np.zeros((len(ref) + 1, len(hyp) + 1))
    
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[i][j] = j  # Insertions
            elif j == 0:
                d[i][j] = i  # Deletions
            else:
                cost = 0 if ref[i-1] == hyp[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1,   # Deletion
                              d[i][j-1] + 1,   # Insertion
                              d[i-1][j-1] + cost)  # Substitution

    # CER is the number of edits divided by the number of characters in the reference
    return d[len(ref)][len(hyp)] / len(ref)


# Calculate CER and WER
# for Case sensitive
llama_cer, llama_wer = 0, 0
qwen_cer, qwen_wer = 0, 0
donut_cer, donut_wer = 0, 0
tr_cer, tr_wer = 0, 0
deep_cer, deep_wer = 0, 0

for ref, qwen, llama, donut, tr, ds in zip(ground_truth_preds, qwen_ocr_preds, llama_ocr_preds, donut_ocr_preds, tr_ocr_preds, deepseek_preds):
    # for llama
    llama_cer += cer(str(ref).strip(), str(llama).strip())
    llama_wer += wer(str(ref).strip(), str(llama).strip())

    # for Qwen
    qwen_cer += cer(str(ref).strip(), str(qwen).strip())
    qwen_wer += wer(str(ref).strip(), str(qwen).strip())

    # for Donut
    donut_cer += cer(str(ref).strip(), str(donut).strip())
    donut_wer += wer(str(ref).strip(), str(donut).strip())
    
    # For tr_ocr
    tr_cer += cer(str(ref).strip(), str(tr).strip())
    tr_wer += wer(str(ref).strip(), str(tr).strip())

    # For deepseek
    deep_cer += cer(str(ref).strip(), str(ds).strip())
    deep_wer += wer(str(ref).strip(), str(ds).strip())
    

# for Case Insensitive
llama_cer, llama_wer = 0, 0
qwen_cer, qwen_wer = 0, 0
donut_cer, donut_wer = 0, 0
tr_cer, tr_wer = 0, 0
deep_cer, deep_wer = 0, 0

for ref, qwen, llama, donut, tr, ds in zip(ground_truth_preds, qwen_ocr_preds, llama_ocr_preds, donut_ocr_preds, tr_ocr_preds, deepseek_preds):
    # for llama
    llama_cer += cer(str(ref).strip().lower(), str(llama).strip().lower())
    llama_wer += wer(str(ref).strip().lower(), str(llama).strip().lower())

    # for Qwen
    qwen_cer += cer(str(ref).strip().lower(), str(qwen).strip().lower())
    qwen_wer += wer(str(ref).strip().lower(), str(qwen).strip().lower())

    # for Donut
    donut_cer += cer(str(ref).strip().lower(), str(donut).strip().lower())
    donut_wer += wer(str(ref).strip().lower(), str(donut).strip().lower())
    
    # For tr_ocr
    tr_cer += cer(str(ref).strip().lower(), str(tr).strip().lower())
    tr_wer += wer(str(ref).strip().lower(), str(tr).strip().lower())
    
    # For deepseek
    deep_cer += cer(str(ref).strip().lower(), str(ds).strip().lower())
    deep_wer += wer(str(ref).strip().lower(), str(ds).strip().lower())

# ASR Baseline
df = pd.read_csv('df_new.csv')

df_ = df.dropna(subset=['Ground Truth', 'TR-OCR Prediction', 'whisper_male'])

trimmed_gt = list(df_['Trimmed Ground Text'])
asr_f = list(df_['whisper_female'])
asr_m = list(df_['whisper_male'])

cer_m, wer_m = 0, 0
cer_f, wer_f = 0, 0

for gt, asr_m_, asr_f_ in zip(trimmed_gt, asr_m, asr_f):
    cer_m += cer(str(gt).strip(), str(asr_m).strip())
    wer_m += wer(str(gt).strip(), str(asr_m).strip())

    cer_f += cer(str(gt).strip(), str(asr_f).strip())
    wer_f += wer(str(gt).strip(), str(asr_f).strip())




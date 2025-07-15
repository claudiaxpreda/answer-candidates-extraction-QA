
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MultiLabelBinarizer

import pandas as pd
import ast
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM, TFDebertaV2Model

MAX_LEN_SEQ = 512
MAX_LEN_BERT = 768
HUGGING_TOKEN = ''


def rectangular(n):
    lengths = {len(i) for i in n}
    return len(lengths) == 1

def split_in_chunks(l, n):
  x = [l[i:i + n] for i in range(0, len(l), n)] 
  return x

def get_max_len(nested_list): 
    return len(max(nested_list, key=len))

def get_pad_len(postitions): 
  max_len = -1
  for pos in postitions:
    print('start: {}, end: {}\n'.format(pos[0], pos[1]))
    if pos[1] - pos[0] + 1 > max_len:
      max_len = pos[1] - pos[0] + 1
  
  return max_len
 
def extractDigits(lst):
    return [[el] for el in lst]

def prepare_input(slice, tokenizer, max_len_pad_set, mlb):
  list_contexts = []
  context_masks = []
  labels = []
  label_score = []

  grouped_slice = slice.groupby('context', sort=False)
  max_len_pad =grouped_slice.count().max()['token_indexes']
  max_len_pad = max_len_pad_set if max_len_pad > max_len_pad_set else max_len_pad

  for group_name, df_group in grouped_slice:
    current_labels = df_group['score_label'].to_list()
    current_labels = split_in_chunks(current_labels, max_len_pad) 
    labels.extend(current_labels)

    mask_list_entry = []

    for pos in df_group['token_indexes'].to_list():
      mask = [0] * MAX_LEN_SEQ
      if pos[0] != pos[1] and pos[1] > 0 and pos[0] > 0:
        mask[pos[0]:pos[1]] = [1] * (pos[1] - pos[0])
      else: 
        mask[pos[0]] = 1
      mask_list_entry.append(mask)

    mask_list_entry = split_in_chunks(mask_list_entry, max_len_pad)
    list_contexts.extend([group_name for i in range(len(mask_list_entry))])

    if len(mask_list_entry[-1]) < max_len_pad: 
      padding = [[0]*MAX_LEN_SEQ for x in range(max_len_pad - len(mask_list_entry[-1]))]
      mask_list_entry[-1].extend(padding)
    
    context_masks.extend(mask_list_entry)
  
  context_masks_tf = tf.convert_to_tensor(context_masks)

  labels_mlb = []
  for entry in labels:  
    current_len = len(entry)
    
    if current_len < max_len_pad:
      entry.extend([0 for x in range(max_len_pad - current_len)])
   
    entry = extractDigits(entry)
    labels_mlb.append(mlb.fit_transform(entry))

  labels_tf =  tf.convert_to_tensor(labels_mlb)
  print('Labesl shape: {}\n'.format(labels_tf.shape))

  # Size (B, 512)
  inputs=tokenizer(list_contexts, padding="max_length", truncation=True, max_length=512, return_tensors='tf')
  context_ids = inputs['input_ids']
  attention_mask =inputs['attention_mask']

  inputs_tf = [
      context_ids,
      attention_mask,
      context_masks_tf
    ]
  
  return inputs_tf, labels_tf , grouped_slice

def prepare_input_with_split(input_df, tokenizer, max_len_pad, mlb):
  train_df, val_df = train_test_split(input_df, random_state=None, test_size=0.2) 

  print(train_df.shape, val_df.shape)

  x_train, y_train, _ = prepare_input(train_df, tokenizer, max_len_pad, mlb)
  x_val, y_val, _ = prepare_input(val_df, tokenizer, max_len_pad, mlb)

  return x_train, y_train, x_val, y_val

def main():
  test_dataset = pd.read_csv('fairytale_dataset/final/qloss_final_train_labeled.csv', 
            keep_default_na=False, converters={"indexes": ast.literal_eval,
            "token_indexes": ast.literal_eval})

  tokenizer=AutoTokenizer.from_pretrained('microsoft/deberta-v3-base', token=HUGGING_TOKEN)

  mlb = mlb = MultiLabelBinarizer(classes=[1, 2, 3, 4])

  prepare_input(test_dataset.iloc[:150], tokenizer, 100, mlb)

if __name__ == "__main__":
    main()
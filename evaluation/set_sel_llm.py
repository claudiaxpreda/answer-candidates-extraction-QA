import ast
import transformers
import sys
import torch
import json
import itertools

import numpy as np
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_PATH  = ''
DESTINATION_PATH = ''
HUGGING_TOKEN = ''

MODEL_NAME_AGEN = ''
MODEL_NAME_QGEN = ''
MAX_LEN_SEQ = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()

def normalize(entry):
  entry = entry.replace("\n", " ").strip()
  entry = entry.replace("\r", " ").strip()
  entry = entry.strip()
  return entry

def ag_prompt_generate(entry):
    return f"Select an answer from the context that can be used to generate a question.\nContext: {entry}\n### Response:"

def gq_prompt_generate(entry):
    return f"Generate a question based on the context and the answer.\nContext: {entry['context']}\nAnswer: {entry['answer_2']}\n### Response:"

def qgen_split(prompt, tokenizer, model):

  model_inputs =  tokenizer(prompt, return_tensors="pt", padding='longest', truncation=True, max_length=2048*2).to(device)
  
  generated_ids = model.generate(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=300,
    pad_token_id=128002,
  )

  generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
  ]
  
  decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
  decoded = [normalize(decode) for decode in decoded]
  return decoded

df_class = pd.read_csv(DATASET_CLASS)
model_agen = AutoModelForCausalLM.from_pretrained(MODEL_NAME_AGEN, token=HUGGING_TOKEN).to("cuda")
tokenizer_agen = AutoTokenizer.from_pretrained(MODEL_NAME_AGEN, padding_side="left", token=HUGGING_TOKEN)
tokenizer_agen.pad_token_id = 128002

# For each top 10, generate an asnwer with Llama
dataset = []
for key, group in df_class.groupby(['context'], sort=False):
    gp = group.sort_values(
      ['label_pred', 'prob_score'], ascending=[True, False]).drop_duplicates(
        ['sequence'], keep='first').head(10)
    
    model_ans = gp['sequence'].to_list()
    model_qs = gp['question_agen'].to_list()
    prompt = ag_prompt_generate(key[0])
    prompts_asel = [prompt] * 10
   
    model_inputs = tokenizer_agen(
      prompts_asel, return_tensors="pt", padding='longest',
      truncation=True, max_length=2048*2).to("cuda")
    
    predictions = model_agen.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=300,
        pad_token_id=128002,
    ) 

    to_decode = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, predictions)
    ]

    decoded = tokenizer_agen.batch_decode(to_decode , skip_special_tokens=True)
    decoded = [normalize(decode) for decode in decoded]
    
    contexts = [key[0]]*10

    pairs = list(zip(contexts, model_ans, model_qs, decoded))
    dataset += pairs


df = pd.DataFrame.from_records(dataset,
      columns=['context', 'answer_1','question_1','answer_2'])

# For each Llama genrated candidate, generate a question. 
tokenizerQ = AutoTokenizer.from_pretrained(
  MODEL_NAME_QGEN, token=HUGGING_TOKEN, padding_side="left")

modelQ = AutoModelForCausalLM.from_pretrained(
  MODEL_NAME_QGEN, token=HUGGING_TOKEN, device_map="auto", 
  torch_dtype=torch.bfloat16)
  
df['prompt'] = df.progress_apply(lambda e: gq_prompt_generate(e), axis=1)
prompts = df['prompt'].to_list()
batch_size = 64

generated_questions = []
for i in tqdm(range(0, len(prompts), batch_size)):
    end_interval = min(i+batch_size, len(prompts))
    inputs = prompts[i:end_interval]
    questions = qgen_split(inputs, tokenizerQ, modelQ)
    generated_questions += questions

df['question_2'] = generated_questions
df.drop(columns=['prompt'], inplace=True)
df.to_csv(DESTINATION_PATH, index=False)

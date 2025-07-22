import os
import random
import sys
import time 
import torch

import pandas as pd
import numpy as np
from tqdm import tqdm
from prompts import JUDGE_PROMPT_2, JUDGE_PROMPT
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "meta-llama/Llama-3.3-70B-Instruct"
HUGGING_FACE_TOKEN=''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()

def normalize(entry):
  entry = entry.replace("\n", " ").strip()
  entry = entry.replace("\r", " ").strip()
  entry = entry.strip()
  return entry

def get_prompt(e): 
    if (e.order == 1): 
        return JUDGE_PROMPT.format(
            context=e.context, 
            question1=e.question_1, 
            answer1=e.answer_1, 
            question2=e.question_2, 
            answer2=e.answer_2)
    else: 
        return JUDGE_PROMPT.format(
            context=e.context, 
            question1=e.question_2, 
            answer1=e.answer_2, 
            question2=e.question_1, 
            answer2=e.answer_1
        )

def extract_judge_verditct(answer):
  try:
    if "Verdict: " in answer:
      verdict = answer.split('Verdict: ')[1]
      verdict = verdict.split('Evaluation: ')[0].strip()
      if (verdict == 'Win' or verdict == 'Lose' or verdict == 'Tie'):
        return verdict 
      
      return None
    
    else:
      return None
  except Exception as e:
    print(e)
    return None

def judge_split(prompt, tokenizer, model):

  model_inputs =  tokenizer(
    prompt, return_tensors="pt", padding='longest', 
    truncation=True, max_length=2048*2).to(device)
  
  generated_ids = model.generate(
    input_ids=model_inputs.input_ids,
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=300,
    pad_token_id=128002,
  )

  generated_ids = [
    output_ids[
      len(input_ids):] for input_ids, output_ids in 
      zip(model_inputs.input_ids, generated_ids)
  ]
  
  decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
  decoded = [normalize(decode) for decode in decoded]
  return decoded

dataset_path = str(sys.argv[1])
destination = str(sys.argv[2])

source_dataset = pd.read_csv(dataset_path).sample(frac=1)

tokenizer = AutoTokenizer.from_pretrained(
  PATH, token=HUGGING_FACE_TOKEN, padding_side="left")

model = AutoModelForCausalLM.from_pretrained(
  PATH, token=HUGGING_FACE_TOKEN, device_map="auto",
  torch_dtype=torch.bfloat16)

np.random.seed(42)
source_dataset['order']  = np.random.choice([1, 2], source_dataset.shape[0])
source_dataset['prompt'] = source_dataset.progress_apply(
    lambda e : get_prompt(e), axis=1)

prompts = source_dataset['prompt'].to_list()
batch_size = 20
generated_feedback = []

for i in tqdm(range(0, len(prompts), batch_size)):
    end_interval = min(i+batch_size, len(prompts))
    inputs = prompts[i:end_interval]
    feedback = judge_split(inputs, tokenizer, model)
    generated_feedback += feedback

source_dataset['full_feedback'] = generated_feedback
source_dataset['verdict'] = source_dataset.progress_apply(
  lambda e : extract_judge_verditct(e.full_feedback), axis=1) 
source_dataset.drop(columns=['prompt'], inplace=True)

source_dataset.to_csv(destination, index=False)

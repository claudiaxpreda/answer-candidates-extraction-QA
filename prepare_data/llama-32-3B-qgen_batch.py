import torch
import torch.nn as nn
import sys
import time 

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# adding Folder_2 to the system path
sys.path.insert(0, '../finetune')
from prepare_dataset_llama import gq_prompt_generate

HUGGING_FACE_TOKEN=''

MAX_LEN_SEQ = 512
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tqdm.pandas()


def normalize(entry):
  entry = entry.replace("\n", " ").strip()
  entry = entry.replace("\r", " ").strip()
  entry = entry.strip()
  return entry

def read_slice(source_file): 
    slice_df = pd.read_csv(source_file, keep_default_na=False)
    return slice_df

def write_slice(df, destination_file): 
  df.to_csv(destination_file, index=False)

def qgen_split(prompt, tokenizer, model):

  model_inputs =  tokenizer(
    prompt, return_tensors="pt",
    padding='longest',
    truncation=True,
    max_length=2048*2
    ).to(device)
  
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


def apply_qgen(model_name, source_dataset):

  tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN, padding_side="left")
  model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGING_FACE_TOKEN, device_map="auto", torch_dtype=torch.bfloat16)
  
  source_dataset['prompt'] = source_dataset.progress_apply(lambda e: gq_prompt_generate(e), axis=1)
  prompts = source_dataset['prompt'].to_list()
  batch_size = 64

  generated_questions = []
  for i in tqdm(range(0, len(prompts), batch_size)):
    end_interval = min(i+batch_size, len(prompts))
    inputs = prompts[i:end_interval]
    questions = qgen_split(inputs, tokenizer, model)
    generated_questions += questions

  source_dataset['question_agen'] = generated_questions
  source_dataset.drop(columns=['prompt'], inplace=True)
 
  return source_dataset


def main(start, end, slice):
  print('Program start time:'+ time.strftime("%H:%M:%S", time.localtime()))

  info_session = f'Slice: {slice} // start: {start} // end: {end} \n'
  print(info_session)
  
  model_name='repo/llama32-3b_qgen_ft_v2'

  filename = f'fairytale_dataset/remastered/ft_og_{slice}_tokens.csv'
  destination = 'fairytale_dataset/set1/qgen/' + 'ft_qgen_{}_'.format(slice) + str(end) +'.csv'

  source_dataset = read_slice(filename)

  if end == 'end':
    source_dataset = source_dataset.iloc[int(start):]
  else: 
    source_dataset = source_dataset.iloc[int(start):int(end)]

  dataset = apply_qgen(model_name, source_dataset)

  write_slice(dataset, destination)

  print('Program end time:'+ time.strftime("%H:%M:%S", time.localtime()))

  return 0

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
import sys
import time 
import torch

import pandas as pd

from transformers import AutoTokenizer, AutoModelForCausalLM
from prepare_dataset_llama import qa_prompt_generate, qa_prompt_learn_processed


HUGGING_FACE_TOKEN=''
MAX_LEN_SEQ = 512
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


def qloss_func(entry, tokenizer, model):
  prompt_with_label = qa_prompt_learn_processed(entry)
  prompt = qa_prompt_generate(entry)
  
  input_prompt = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
  whole_prompt = tokenizer(prompt_with_label, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)

  #Add the eos token to the whole prompts, for the whole batch
  whole_prompt['input_ids'] = torch.cat((whole_prompt['input_ids'], torch.tensor([[tokenizer.eos_token_id]] * whole_prompt['input_ids'].shape[0], device=device)), dim=1)
  whole_prompt['attention_mask'] = torch.cat((whole_prompt['attention_mask'], torch.tensor([[1]] * whole_prompt['attention_mask'].shape[0], device=device)), dim=1)
  
  outputs = model(input_ids=whole_prompt['input_ids'], attention_mask=whole_prompt['attention_mask'])
  logits = outputs.logits[0]

  whole = whole_prompt['input_ids'][0]
  inputp = input_prompt['input_ids'][0]
  padding = torch.count_nonzero(whole == tokenizer.pad_token_id)
  whole = whole[padding:]
  padding = torch.count_nonzero(inputp == tokenizer.pad_token_id)
  inputp = inputp[padding:]

  # Remove the last logit (unnecessary, automatically added by the model)
  logits = logits[:-1]

  # Get from the logits just the ones corresponding to the actual generation (label)
  good_logit = logits[-(len(whole) - len(inputp)):]

  # Get the label
  good_label = whole[len(inputp):]


  loss = loss_fn(
      good_logit,
      good_label,
  )

  return loss.item()


def apply_qloss(model_name, source_dataset):
  tokenizer = AutoTokenizer.from_pretrained(model_name, HUGGING_FACE_TOKEN, padding_side="left")
  model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGING_FACE_TOKEN, device_map="auto", torch_dtype=torch.bfloat16)
  tokenizer.pad_token_id = 128002

  source_dataset['loss'] = source_dataset.apply(
    lambda e : qloss_func(e, tokenizer, model), axis=1) 
  
  source_dataset['rloss'] = source_dataset.apply(
    lambda e : round(e['loss'], 4), axis=1) 
 
  return source_dataset


def main(source, destination, slice):
  print('Program start time:'+ time.strftime("%H:%M:%S", time.localtime()))
  
  filename = 'fairytale_dataset/QG/{}/'.format(slice) + source
  destination = 'fairytale_dataset/QAL/{}/'.format(slice) + destination

  model_name='repo/llama32-3b_qa_ft'

  source_dataset = read_slice(filename)
  #source_dataset = source_dataset.iloc[0:10]
  dataset = apply_qloss(model_name, source_dataset)

  write_slice(dataset, destination)

  print('Program end time:'+ time.strftime("%H:%M:%S", time.localtime()))

  return 0

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
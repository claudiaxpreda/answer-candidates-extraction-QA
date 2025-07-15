import sys
import time 
import torch

import pandas as pd
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, '../finetune')

from prepare_dataset_llama import qa_prompt_generate, qa_prompt_learn_processed


HUGGING_FACE_TOKEN=''
MAX_LEN_SEQ = 512
loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')


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


def qloss_func(prompts_with_label, prompts, tokenizer, model):

  input_prompt = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)
  whole_prompt = tokenizer(prompts_with_label, return_tensors="pt", truncation=True, max_length=2048, padding='longest').to(device)

  #Add the eos token to the whole prompts, for the whole batch
  whole_prompt['input_ids'] = torch.cat((whole_prompt['input_ids'], torch.tensor([[tokenizer.eos_token_id]] * whole_prompt['input_ids'].shape[0], device=device)), dim=1)
  whole_prompt['attention_mask'] = torch.cat((whole_prompt['attention_mask'], torch.tensor([[1]] * whole_prompt['attention_mask'].shape[0], device=device)), dim=1)
  
  with torch.no_grad():
    outputs = model(input_ids=whole_prompt['input_ids'], attention_mask=whole_prompt['attention_mask'])
    logits = outputs.logits

  batch_losses = []
  for logit, input, whole in zip(logits, input_prompt['input_ids'], whole_prompt['input_ids']):
    # Remove padding
    padding = torch.count_nonzero(whole == tokenizer.pad_token_id)
    whole = whole[padding:]
    padding = torch.count_nonzero(input == tokenizer.pad_token_id)
    input = input[padding:]

    # Remove the last logit (unnecessary, automatically added by the model)
    logit = logit[:-1]

    # Get from the logits just the ones corresponding to the actual generation (label)
    good_logit = logit[-(len(whole) - len(input)):]

    # Get the label
    good_label = whole[len(input):]

    loss = loss_fn(
        good_logit,
        good_label,
    )

    batch_losses.append(loss.item())

  return batch_losses


def apply_qloss(model_name, source_dataset):
  tokenizer = AutoTokenizer.from_pretrained(model_name, HUGGING_FACE_TOKEN, padding_side="left")
  model = AutoModelForCausalLM.from_pretrained(model_name, token=HUGGING_FACE_TOKEN, device_map="auto", torch_dtype=torch.bfloat16)
  tokenizer.pad_token_id = 128002

  source_dataset['prompt_with_label'] = source_dataset.progress_apply(lambda e: qa_prompt_learn_processed(e), axis=1)
  prompts_with_label = source_dataset['prompt_with_label'].to_list()

  source_dataset['prompt'] = source_dataset.progress_apply(lambda e:qa_prompt_generate(e), axis=1)
  prompts = source_dataset['prompt'].to_list()
  
  batch_size = 64

  losses = []
  for i in tqdm(range(0, len(prompts), batch_size)):
    end_interval = min(i+batch_size, len(prompts))
    inputs_label = prompts_with_label[i:end_interval]
    inputs = prompts[i:end_interval]
    loss_item = qloss_func(inputs_label, inputs, tokenizer, model)
    losses += loss_item
  
  source_dataset['loss'] = losses
  
  source_dataset['rloss'] = source_dataset.apply(
    lambda e : round(e['loss'], 4), axis=1) 
  
  source_dataset.drop(columns=['prompt_with_label', 'prompt'], inplace=True)
  return source_dataset


def main(source, slice):
  print('Program start time:'+ time.strftime("%H:%M:%S", time.localtime()))
  print (f'Index: {source} // slice: {slice} \n')
  
  filename = 'fairytale_dataset/set1/qgen/ft_qgen_{}_{}.csv'.format(slice, source)
  destination = 'fairytale_dataset/set1/qloss/ft_qloss_{}_{}.csv'.format(slice, source)

  model_name='repo/llama32-3b_qa_ft-v3'

  source_dataset = read_slice(filename)
  #source_dataset = source_dataset.iloc[0:10]
  dataset = apply_qloss(model_name, source_dataset)

  write_slice(dataset, destination)

  print('Program end time:'+ time.strftime("%H:%M:%S", time.localtime()))

  return 0

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
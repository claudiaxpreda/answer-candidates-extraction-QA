import pandas as pd
from transformers import DebertaV2TokenizerFast

MAX_LEN_SEQ = 512
splits = ['test', 'validation', 'train']

def get_bert_token_indexes(context, pair_indx, seq, tokenizer, debug=True, level='d', space_p=True):
  start=-1
  end=-1
  contor=0
  
  tokenscon = tokenizer(context, padding="max_length", truncation=True, max_length=MAX_LEN_SEQ,  return_offsets_mapping=True)

  for token_id, pos in zip(tokenscon['input_ids'], tokenscon['offset_mapping']):
  
    if space_p and pos[0] != 0: 
      pos_0 = pos[0] + 1 
    else:
      pos_0 = pos[0]
    
    if debug  and level == 'v':
      print(token_id, pos, context[pos[0]:pos[1]])  

    if pos_0 == pair_indx[0] and pos[1] == pair_indx[1] and token_id != 1 and token_id != 2 and token_id != 0 and token_id != 50264:
      start = contor
      end = contor
      return (start, end)

    if pos_0 == pair_indx[0] and pair_indx[1] > pos[1] and token_id != 1 and token_id != 2 and token_id != 0 and token_id != 50264:
      start = contor

    if pos_0 > pair_indx[0]  and pos[1] == pair_indx[1] and token_id != 1 and token_id != 2 and token_id != 0 and token_id != 50264:
      end = contor + 1

    contor += 1

  if debug  and (level == 'n' or level == 'v'):
      print('== Current sequnece ==\n')
      print(seq)
  
  if debug and level == 'v':
    print('== Token indexes ==\n')
    print('Start: {}, End: {}'.format(start, end))
    print('== Char indexes ==\n')
    print('Start: {}, End: {}'.format(pair_indx[0], pair_indx[1]))

  check_string = tokenizer.decode(tokenscon['input_ids'][start:end])

  if debug and (level == 'n' or level == 'v'):
      print('== Current sequnece ==\n')
      print(check_string)
  
  if start == -1 or end == -1: 
    start = - 1
    end = - 1
  
  return (start, end)

def compute_token_indexes(source_dataset, tokenizer):
  source_dataset['token_indexes'] = source_dataset.apply(
    lambda e : get_bert_token_indexes(e['context'], (e['start_char'], e['end_char']), e['sequence'], tokenizer), axis=1) 
  
  indx_to_drop=source_dataset[source_dataset.token_indexes == (-1,-1)].index
  
  if len(indx_to_drop == 0):
    print('To delete ' + str(indx_to_drop))
    print(len(indx_to_drop))
    source_dataset = source_dataset.drop(indx_to_drop)

  return source_dataset


model_name='microsoft/deberta-v3-base'
tokenizer=DebertaV2TokenizerFast.from_pretrained(model_name, add_prefix_space=True)


for split in splits:
  print(split)
  path_to_load = f'../fairytale_dataset/remastered/ft_og_{split}.csv'
  path_to_save = f'../fairytale_dataset/remastered/ft_og_{split}_tokens.csv'
  source_dataset = pd.read_csv(path_to_load, keep_default_na=False)
  df_dataset = compute_token_indexes(source_dataset, tokenizer)
  df_dataset.to_csv(path_to_save, index=False)
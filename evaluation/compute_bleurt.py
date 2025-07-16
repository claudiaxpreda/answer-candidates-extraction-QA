import ast
import itertools
import json
import networkx as nx
import numpy as np
import pandas as pd
import sys
import transformers
import torch


from tqdm import tqdm
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer

HUGGING_TOKEN = ''
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20').to(device)
tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

# Paths for the test datasets
DATASET_CLASS = ''
DATASET_NER = ''
DATASET_LLAMA = ''

df_llama = pd.read_csv(DATASET_LLAMA)
df_ner = pd.read_csv(DATASET_NER)
df_class = pd.read_csv(DATASET_CLASS)

# All the edges of type (test_candidates, target)
def create_edges(labels, targets): 
  refs = []
  cands = []  

  for (l, t) in list(itertools.product(labels, targets)):
    refs.append(l)
    cands.append(t)
    with torch.no_grad():
      inputs = tokenizer(refs, cands, padding='longest', return_tensors='pt').to('cuda')
      res = model(**inputs).logits.flatten().tolist()
  
  edges = list(zip(refs, cands, res))

  return edges

df_llama_gp = df_llama.groupby(['context'])

res = []
for key, group in df_class.groupby(['context'], sort=False):
  gp = group.sort_values(['label_pred','prob_score'], ascending=[True, False]).drop_duplicates(['sequence'], keep='first').head(10)
  sequences = gp['sequence'].to_list()
  seq_gp = df_llama_gp.get_group(key)['gt_answer'].tolist()
  edges = create_edges(seq_gp, sequences)
  G = nx.Graph()
  G.add_weighted_edges_from(edges)
  result = sorted(nx.max_weight_matching(G))
  score = 0
  for (u,v) in result:
    score += G.get_edge_data(u, v)['weight']
  if len(result) > 0: 
    res.append(score)
  else:
    res.append(0)

final_score = sum(res) / len(res)
print('Print score for CLASSIFICATION vs GROUND TRUTH: ' + str(final_score))

res = []
for index, entry in df_ner.iterrows():
  sequences = entry['entities']
  seq_gp = df_llama_gp.get_group((entry['context'],))['gt_answer'].tolist()
  edges = create_edges(seq_gp, sequences)
  G = nx.Graph()
  G.add_weighted_edges_from(edges)
  result = sorted(nx.max_weight_matching(G))
  score = 0
  for (u,v) in result:
    score += G.get_edge_data(u, v)['weight']
  if len(result) > 0: 
    res.append(score)
  else:
    res.append(0)

final_score = sum(res) / len(res)
print('Print score for NER vs GROUND TRUTH: ' + str(final_score))

res = []
for key, group in df_llama.groupby(['context'], sort=False):
  sequences = group['prediction-llama'].to_list()
  seq_gp = df_llama_gp.get_group(key)['gt_answer'].tolist()
  edges = create_edges(seq_gp, sequences)
  G = nx.Graph()
  G.add_weighted_edges_from(edges)
  result = sorted(nx.max_weight_matching(G))
  score = 0
  for (u,v) in result:
    score += G.get_edge_data(u, v)['weight']
  if len(result) > 0: 
    res.append(score)
  else:
    res.append(0)

final_score = sum(res) / len(res)
print('Print score for LLAMA AGEN vs GROUND TRUTH: ' + str(final_score))
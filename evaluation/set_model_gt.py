import ast
import pandas as pd
import transformers
import numpy as np
import sys
import torch
import json
from tqdm import tqdm
import itertools

from transformers import AutoTokenizer, AutoModelForCausalLM

DATASET_PATH = ''
DATASET_GT_PATH = ''
DESTINATION_PATH = ''


df_class = pd.read_csv(DATASET_PATH)
df_og =  pd.read_csv(DATASET_GT_PATH)
df_og_gp = df_og.groupby(['context'], sort=False)

dataset = []

for key, group in df_class.groupby(['context'], sort=False):
    gp = group.sort_values(['label_pred','prob_score'], ascending=[True, False]).drop_duplicates(['sequence'], keep='first').head(10)
    model_ans = gp['sequence'].to_list()
    model_qs = gp['question_agen'].to_list()
    context = [key[0]]*len(model_ans)
    model_pairs = list(zip(context, model_ans, model_qs))

    og_gp = df_og_gp.get_group(key)
    og_ans = og_gp['gt_answer'].to_list()
    og_qs = og_gp['gt_question'].to_list()
    og_pairs = list(zip(og_ans, og_qs))

    product = list(itertools.product(model_pairs, og_pairs))
    product_joined = []

    for (e1, e2) in product: 
        e = e1 + e2
        product_joined.append(e)

    dataset += product_joined

df = pd.DataFrame.from_records(dataset, columns=['context', 'answer_1','question_1','answer_2', 'question_2'])
df.to_csv(DESTINATION_PATH, index=False)



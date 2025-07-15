from tqdm import tqdm
import re
import pandas as pd
import sys
import ast
import spacy

from tqdm import tqdm
tqdm.pandas()

en = spacy.load('en_core_web_md')
sw_spacy = en.Defaults.stop_words

VGOOD = 1
GOOD = 2
AVERAGE = 3 
BAD = 4

scores_dict = {
    "1_start": 0, "1_end": 1, 
    "2_start": 1, "2_end": 2,
    "3_start": 2, "3_end": 5,
    "4_start": 5
    }

scores_dict_new = {
    "1_start": 0, "1_end": 10, 
    "2_start": 10, "2_end": 13,
    "3_start": 13, "3_end": 15,
    "4_start": 15
    }

def assign_label_entry(entry):
    for l in [VGOOD, GOOD, AVERAGE]:
        start = "{}_start".format(l)
        end = "{}_end".format(l)
        if entry >= scores_dict_new[start] and entry < scores_dict_new[end]: 
            return l

    return BAD

def is_stopword(entry): 
    entry = entry.strip()
    entry = entry.lower()

    if entry in sw_spacy: 
        return True 
    else:
        return False


SPLITS = ['test', 'train', 'validation']

for split in SPLITS: 
    source = f'fairytale_dataset/final/{split}_all.csv'
    destination = f'fairytale_dataset/final/{split}_labeled.csv'

    print(f'Processing split: {split}\n')
    df = pd.read_csv(source, keep_default_na=False, converters={"indexes": ast.literal_eval})
    df['score_label'] = df['loss'].progress_apply(lambda e : assign_label_entry(e))
    df.to_csv(destination, index=False)
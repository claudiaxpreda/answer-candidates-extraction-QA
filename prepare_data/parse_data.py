from tqdm import tqdm
import re
import spacy
import benepar
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from string import punctuation


benepar.download('benepar_en3')
nlp = spacy.load("en_core_web_md")
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

def prepare_slice(slice):
    columns = ["story_section", "question", "answer1", "ex-or-im"]
    df = pd.DataFrame(slice, columns=columns)
    df.rename(columns={"story_section": "context", 
        "question": "gt_question", "answer1": "gt_answer"}, inplace=True)
    
    df["context"] = df["context"].map(lambda x: re.sub(r'\s([?.!,;"](?:\s|$))', r'\1', x))
    df["context"] = df["context"].map(lambda x: re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), x))
    df['context'] = df['context'].apply(lambda e : e.replace("\n", " ").strip())
    df['context'] = df['context'].apply(lambda e : e.replace("\r", " ").strip())
   
    df["gt_question"] = df["gt_question"].map(lambda x: re.sub(r'\s([?.!,;"](?:\s|$))', r'\1', x))
    df["gt_question"] = df["gt_question"].map(lambda x: re.sub("(^|[.?!])\s*([a-zA-Z])", lambda p: p.group(0).upper(), x))
    
    df.drop_duplicates(inplace=True, ignore_index=True)
    return df

def create_entry(context, entry):
  seq, start_char, end_char, labels = entry
  return {'context': context, 'sequence': seq, 'start_char': start_char, 'end_char': end_char, 'labels': labels}

def dfs(node, texts):
    texts.append((node.text, node.start_char, node.end_char, node._.labels))
    for child in node._.children:
        dfs(child, texts)

def get_syntax_tree_nodes(sentence):
    doc = nlp(sentence)
    nodes = list(doc.sents)
    texts = []

    for node in nodes: 
      dfs(node, texts)
    
    texts = [ t for t in texts if t[0] not in punctuation]
    return texts

def create_dataset_slice(df):
  dict_list = []
  for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    context = row['context']
    tree = get_syntax_tree_nodes(context)
    for entry in tree:
      dict_entry = create_entry(context, entry)
      dict_list.append(dict_entry)
  
  return pd.DataFrame.from_records(dict_list)



slices = ['validation', 'train', 'test']
root_dir = '../fairytale_dataset/remastered/'
fairytale_dataset = load_dataset('WorkInTheDark/FairytaleQA')

for entry in slices: 
    entry_slice = fairytale_dataset[entry]
    entry_data = prepare_slice(entry_slice)
    name_file_slice = root_dir + 'ft_og_{}_noparse.csv'.format(entry)
    entry_data.to_csv(name_file_slice, index=False)
    print("Finished processing slice: {}".format(entry))

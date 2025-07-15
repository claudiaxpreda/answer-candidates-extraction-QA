import pandas as pd
from datasets import load_dataset

DATASET_FT= 'WorkInTheDark/FairytaleQA'
TRAIN_FILE = 'fairytale_dataset/ft_og_train_noparse.csv'
VALIDATION_FILE = 'fairytale_dataset/ft_og_validation_noparse.csv'
TEST_FILE = 'fairytale_dataset/ft_og_test_noparse.csv'

QUESTION_FIELD = 'gt_question'
ANSWER_FIELD = 'gt_answer'
QUESTION_FIELD = 'question_agen'

def normalize(entry):
  entry = entry.replace("\n", " ").strip()
  entry = entry.replace("\r", " ").strip()
  return entry

def qg_prompt_learn(entry):
    return f"Generate a question based on the context and the answer.\nContext: {entry['context']}\nAnswer: {entry[ANSWER_FIELD]}\n### Response: {entry[QUESTION_FIELD]}"

def qa_prompt_learn(entry):
    return f"Answer the following question based on the context.\nContext: {entry['context']}\nQuestion: {entry[QUESTION_FIELD]}\n### Response: {entry[ANSWER_FIELD]}"

def ag_prompt_learn(entry): 
    return f"Select an answer from the context that can be used to generate a question.\nContext: {entry['context']}\n### Response: {entry[ANSWER_FIELD]}"

def qa_prompt_learn_processed(entry):
    return f"Answer the following question based on the context.\nContext: {entry['context']}\nQuestion: {entry[QUESTION_FIELD]}\n### Response: {entry['sequence']}"

def gq_prompt_generate(entry):
    return f"Generate a question based on the context and the answer.\nContext: {entry['context']}\nAnswer: {entry['sequence']}\n### Response:"

def  qa_prompt_generate(entry):
    return f"Answer the following question based on the context.\nContext: {entry['context']}\nQuestion: {entry[QUESTION_FIELD]}\n### Response:"

def ag_prompt_generate(entry):
    return f"Select an answer from the context that can be used to generate a question.\nContext: {entry['context']}\n### Response:"

def get_input_qg(split, csv=True, token=''): 
    if csv == True:
        fairytale_dataset = load_dataset("csv", data_files={'train': TRAIN_FILE, 
            'validation': VALIDATION_FILE, 'test': TEST_FILE})
        fairytale_dataset = fairytale_dataset[split]

    else:
        fairytale_dataset = load_dataset(DATASET_FT, split=split, token=token)
    print(fairytale_dataset)
    fairytale_dataset = fairytale_dataset.map(lambda e : {'context': normalize(e['context'])})

    fairytale_dataset = fairytale_dataset.map(lambda e : {'input_prompt': qg_prompt_learn(e)})

    return fairytale_dataset

def get_input_qa(split, csv=True, token=''): 
    if csv == True:
        fairytale_dataset = load_dataset("csv", data_files={'train': TRAIN_FILE, 
            'validation': VALIDATION_FILE, 'test': TEST_FILE})
        fairytale_dataset = fairytale_dataset[split]
    else:
        fairytale_dataset = load_dataset(DATASET_FT, split=split, token=token)
    print(fairytale_dataset)
    fairytale_dataset = fairytale_dataset.map(lambda e : {'context': normalize(e['context'])})

    fairytale_dataset = fairytale_dataset.map(lambda e : {'input_prompt': qa_prompt_learn(e)})

    return fairytale_dataset

def get_input_ag(split, csv=True, token=''): 
    if csv == True:
        fairytale_dataset = load_dataset("csv", data_files={'train': TRAIN_FILE, 
            'validation': VALIDATION_FILE, 'test': TEST_FILE})
        fairytale_dataset = fairytale_dataset[split]
    else:
        fairytale_dataset = load_dataset(DATASET_FT, split=split, token=token)
    print(fairytale_dataset)
    fairytale_dataset = fairytale_dataset.map(lambda e : {'context': normalize(e['context'])})

    fairytale_dataset = fairytale_dataset.map(lambda e : {'input_prompt': ag_prompt_learn(e)})

    return fairytale_dataset

import itertools
import pandas as pd
import random
import sys
import torch
import spacy 
import nltk
import networkx as nx

nltk.download('punkt_tab')
nltk.download('punkt')

from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from sentence_transformers import SentenceTransformer


HUGGING_TOKEN = "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"
TRESHOLD = 0.3
TOP = 10
TOP5 = 5
DATASET_CLASS = 'fairytale_dataset/set1/results/test_class.csv'
MODEL_SENTENCE = 'sentence-transformers/all-mpnet-base-v2'


sys.path.insert(0, '/export/home/acs/stud/c/claudia.preda2307/qagloss/prepare_data')


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_candidates(context): 
    return

def get_random_entry(path): 
    df_class = pd.read_csv(path)
    random_sample = df_class['context'].to_list()
    random_sample_text = random.choice(random_sample)
    df_class_gp = df_class.groupby(['context'])
    group = df_class_gp.get_group((random_sample_text,))

    return (random_sample_text, group)

def get_sentences_indexes(sentences): 
    index_sentences = {}
    start = 0
    end = 0

    for sent in sentences: 
        end = start + len(sent)
        index_sentences[sent] = (start, end)        
        start = end + 1
    
    return index_sentences

def map_answers_sents(ordered_sentences_top, index_sentences, group_sort):
    answers = []

    for sent in ordered_sentences_top: 
        (start, end) = index_sentences[sent]

        cond_gp = group_sort[ 
            (group_sort['start_char'] >= start) & (group_sort['end_char'] <= end)].reset_index(drop=True)


        cand = cond_gp.head(1)
        answers.append((
                cand['sequence'].to_list()[0], 
                cand['question_agen'].to_list()[0],
                cand['start_char'].to_list()[0],
                cand['end_char'].to_list()[0]
            ))
    
    return answers

def top_k(group, k = TOP): 
    group_sort = group.sort_values(['prob_score'], 
                    ascending=False).drop_duplicates(['sequence'], keep='first')
    
    answers = group_sort.head(k)['sequence'].to_list()
    questions = group_sort.head(k)['question_agen'].to_list()
    start_chars = group_sort.head(k)['start_char'].to_list()
    end_chars = group_sort.head(k)['end_char'].to_list()

    return list(zip(answers, questions, start_chars, end_chars))


def top_k_filtered(group, k = TOP): 
    model = BleurtForSequenceClassification.from_pretrained(
            'lucadiliello/BLEURT-20').to(device)
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')
    
    group = group.loc[group['label_pred'] != 4 ]
    group = group.loc[group['label_pred'] != 3 ]

    group_sort = group.sort_values(['prob_score'], 
                    ascending=False).drop_duplicates(['sequence'], keep='first')
    
    candidates_refs = group_sort.head(TOP)['sequence'].to_list()
    
    ref = random.choice(candidates_refs)

    candidates= group_sort['sequence'].to_list()
    references  = [ref] * len(candidates)

    inputs = tokenizer(
        references, candidates, padding='longest', 
        return_tensors='pt').to('cuda')
    
    scores = model(**inputs).logits.flatten().tolist()
    group_sort['scores'] = scores

    group_select = group_sort.sort_values(
        ['scores'], ascending=True).head(k - 1)
    row = group_sort.loc[group_sort['sequence'] == ref]

    # print("Ref: " + ref) 
    # print(row)

    answers = [ref] + group_select['sequence'].to_list()
    
    questions = row['question_agen'].to_list() + group_select['question_agen'].to_list()
    start_chars = row['start_char'].to_list() + group_select['start_char'].to_list()
    end_chars = row['end_char'].to_list() + group_select['end_char'].to_list()

    return list(zip(answers, questions, start_chars, end_chars))

def get_top_k_sentence_answers(text, group, top=TOP5): 
    model = SentenceTransformer(MODEL_SENTENCE)

    sentences = nltk.sent_tokenize(text)
    index_sentences = get_sentences_indexes(sentences)


    embeddings = model.encode(sentences)
    similarities = model.similarity(embeddings, embeddings).numpy()

    nx_graph = nx.from_numpy_array(similarities)
    scores = nx.pagerank(nx_graph)
   
    scores_sentences={
        sentence:scores[index] for index,sentence in enumerate(sentences)
        }
    
    top_k = max(top, len(sentences))

    ordered_sentences = dict(
        sorted(scores_sentences.items(), key=lambda x: x[1], reverse=True)[:top_k]
        )

    ordered_sentences_top =list(ordered_sentences.keys())

    group_sort = group.sort_values(['prob_score'], 
                ascending=False).drop_duplicates(['sequence'], keep='first')
    

    answers = map_answers_sents(ordered_sentences_top, index_sentences, group_sort)

    return answers

def compute_distance(entry, rstart, rend, len_text):
    ent_start = entry['start_char']
    ent_end = entry['end_char']

    if (ent_start <= rstart) and (rend <= ent_end): 
        return len_text
    
    if (rstart - ent_start - 1 == 0) or (ent_start - rend - 1 == 0): 
        return len_text
    
    if (ent_start < rstart ):
        return len_text / (rstart - ent_start -1)
    
    if (ent_start > rend): 
        return len_text / (ent_start - rend - 1)

    return len_text



def get_kmeans_distance(text, group, top=TOP5): 
    group = group.loc[group['label_pred'] != 4 ]
    group = group.loc[group['label_pred'] != 3 ]

    model = BleurtForSequenceClassification.from_pretrained(
        'lucadiliello/BLEURT-20').to(device)
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20')

    group_sort = group.sort_values(['prob_score'], 
                ascending=False).drop_duplicates(['sequence'], keep='first').reset_index(drop=True)
    
    ref_pd = group_sort.head(1)
    # ref = group_sort.head(1)['sequence'].to_list()[0]
    # end_ref = group_sort.head(1)['end_char'].to_list()[0]
    # start_ref = group_sort.head(1)['start_char'].to_list()[0]
    # question_ref = group_sort.head(1)['question_agen'].to_list()[0]

    len_text = len(text)
    
    #   answers = [ref] + group_select['sequence'].to_list()
        
    #     questions = [question_ref]+ group_select['question_agen'].to_list()
    #     start_chars = [start_ref] + group_select['start_char'].to_list()
    #     end_chars = [end_ref] + group_select['end_char'].to_list()
    refs_pd = [ref_pd]
    
    for _ in range(top - 1):
        mins = []
       
        for df in refs_pd: 
            ref = df['sequence'].to_list()[0]
            end_ref = df['end_char'].to_list()[0]
            start_ref = df['start_char'].to_list()[0]
            question_ref = df['question_agen'].to_list()[0]

        
            group_sort['dist_score'] = group_sort.apply(
                lambda e :  compute_distance(e, start_ref, end_ref, len_text), axis=1)

            candidates= group_sort['sequence'].to_list()
            references  = [ref] * len(candidates)

            inputs = tokenizer(
                references, candidates, padding='longest', 
                return_tensors='pt').to('cuda')
        
            scores = model(**inputs).logits.flatten().tolist()
            group_sort['sim_score'] = scores

            group_sort['final_score'] = group_sort.apply(
                lambda e : (e['sim_score'] + e['dist_score']) / 2, axis=1
            )


            group_sort = group_sort.sort_values(['final_score'], ascending=True)
            mins.append(group_sort.head(1))
            #print(group_sort.head(1))
            #print(df)
        mins_scores = [minv['final_score'].to_list()[0] for minv in mins]
        min_value = min(mins_scores)
        min_index = mins_scores.index(min_value)
        refs_pd.append(mins[min_index])
        #print(len(refs_pd))


    answers = [ref['sequence'].to_list()[0] for ref in refs_pd]
    questions = [ref['question_agen'].to_list()[0] for ref in refs_pd]
    start_chars =[ref['start_char'].to_list()[0] for ref in refs_pd]
    end_chars = [ref['end_char'].to_list()[0] for ref in refs_pd]

    return list(zip(answers, questions, start_chars, end_chars))

text, test = get_random_entry(DATASET_CLASS)


print('========CONTEXT========')
print(text)

results = top_k_filtered(test, TOP5)
print('\n========FILTERED========')
print(results)

print('\n=========TOP_SENTS=========')
nresults = top_k(test)
print(nresults)

print('\n=========TOP_KMEANS=========')
print(get_kmeans_distance(text, test))
import sys
import pandas as pd
import random
from tqdm import tqdm 

SCALE = 400
INIT_RATING = 1500
BASE = 10
K = 32

score_map =  {
  'Win': 1,
  'Lose': 0,
  'Tie': 0.5
}

MODEL_LLM = "MODLLM"
MODEL_GT = "MODGT"
GT_LLM = "GTLLM"
ALL = "ALL"
TEST = "TEST"

path_map = {
  'MODLLM': '',
  'MODGT': '',
  'GTLLM': ''
}

model_score_init = {
    'model': INIT_RATING,
    'gt': INIT_RATING,
    'llm': INIT_RATING
}

def get_model_option(option):
    model_1 = ''
    model_2 = ''

    if option == MODEL_LLM: 
        model_1 = 'model'
        model_2 = 'llm'
    
    if option == MODEL_GT:
        model_1 = 'model'
        model_2 = 'gt'
    
    if option == GT_LLM:
        model_1 = 'gt'
        model_2 = 'llm'
    
    if option == TEST: 
        model_1 = 'model'
        model_2 = 'llm'
    
    return (model_1, model_2)
    

def probability(R1, R2):
    return 1 / (1 + pow(10, ((R1 - R2) / SCALE )))

def elo_compute_step(R1, R2, result): 

    P1 = probability(R2, R1)
    P2 = probability(R1, R2)

    R1 = R1 + K * (result - P1)
    R2 = R2 + K * ((1 - result) - P2)

    return (R1, R2)

def compute_result(order, verdict):
    if order == 1: 
        return  score_map[verdict]
    else:
        return 1 - score_map[verdict]

# Result is always the outcome for the first player
def process_result(data, model_1, model_2): 
    data['result'] = data.apply(lambda x: compute_result(x.order, x.verdict), axis = 1)
    results = data['result'].to_list()
    games = []

    for result in results: 
        games.append((model_1, model_2, result))
    
    return games

def read_data(option): 
    data_path = 'fairytale_dataset/set1/evaluation/' + path_map[option]
    data = pd.read_csv(data_path)

    return data

def elo_compute(data, R1, R2): 
    data['result'] = data.apply(lambda x: compute_result(x.order, x.verdict), axis = 1)
    results = data['result'].to_list()
    
    print(len(results))
    print(results.count(0))
    for result in results: 
        (R1, R2) = elo_compute_step(R1, R2, result)
    
    return (R1, R2)

def elo_compute_battle(): 
    all_games = [] 

    for option in (MODEL_LLM, MODEL_GT, GT_LLM): 
        print(option)
        data = read_data(option)
        (model_1, model_2) = get_model_option(option) 
        games = process_result(data, model_1, model_2)
        all_games += games

    print(len(all_games))
    random.shuffle(all_games)
    
    scores = model_score_init

    for (model_1, model_2, result) in all_games: 
        (R1, R2) = elo_compute_step(scores[model_1], scores[model_2], result)
        print (model_1, model_2, R1, R2)

        scores[model_1] = R1
        scores[model_2] = R2


    return scores


option = str(sys.argv[1])
print(option)

if option != ALL or option == TEST:
    data_path = path_map_op[option]
    data = pd.read_csv(data_path)
    (model_1, model_2) = get_model_option(option) 
    (R1F, R2F) = elo_compute(data, model_score_init[model_1], model_score_init[model_2])
    print('Model 1 - {} - Score: {} \n Model 2 - {} - Score : {}'.format(model_1, R1F, model_2, R2F))
else:
    print(elo_compute_battle())


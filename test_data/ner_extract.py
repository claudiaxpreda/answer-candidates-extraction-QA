import pandas as pd
import spacy 
from tqdm import tqdm

ORGINAL_DATASET_PATH = ''
DESTINATION_TEST_NER = ''

nlp = spacy.load('en_core_web_lg')

og_dataset_test = pd.read_csv(ORGINAL_DATASET_PATH, keep_default_na=False)
og_dataset_test_gp = og_dataset_test.groupby(['context'], sort=False)

keys = [key[0] for key, _ in og_dataset_test_gp]
entries_dict = []

for key in tqdm(keys):
    doc = nlp(key)
    entities = []
    for word in doc.ents:
        entities.append(word.text)

    entries_dict.append({'context': key, 'entities': entities})

final_df = pd.DataFrame.from_records(entries_dict)
final_df.to_csv(DESTINATION_TEST_NER)
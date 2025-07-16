
import sys
import pandas as pd

from tqdm import tqdm
from huggingface_hub import from_pretrained_keras
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, '../finetune')
from prepare_dataset_llama import  ag_prompt_generate

HUGGING_TOKEN = ''
ORGINAL_DATASET_PATH = ''
MODEL_AGEN_NAME = ''
DESTINATION_LLAMA_AGEN = ''

def normalize(entry):
  entry = entry.replace("\n", " ").strip()
  entry = entry.replace("\r", " ").strip()
  entry = entry.strip()
  return entry


og_dataset_test = pd.read_csv(ORGINAL_DATASET_PATH, keep_default_na=False)
og_dataset_test['prompt'] = og_dataset_test.apply(
    lambda e : ag_prompt_generate(e), axis=1) 
prompts = og_dataset_test['prompt'].tolist()

model_agen = AutoModelForCausalLM.from_pretrained(
    MODEL_AGEN_NAME, token=HUGGING_TOKEN).to("cuda")
tokenizer_agen = AutoTokenizer.from_pretrained(
    MODEL_AGEN_NAME, padding_side="left", token=HUGGING_TOKEN)
tokenizer_agen.pad_token_id = 128002

og_dataset_test_agen = []
batch_size = 128

for i in tqdm(range(0, len(prompts), batch_size)):
    end_interval = min(i+batch_size, len(prompts))

    model_inputs = tokenizer_agen(
        prompts[i:end_interval],
        return_tensors="pt", 
        padding='longest',
        truncation=True,
        max_length=2048*2
    ).to("cuda")

    predictions = model_agen.generate(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=300,
        pad_token_id=128002,
    ) 
    to_decode = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, predictions)
    ]

    decoded = tokenizer_agen.batch_decode(to_decode, skip_special_tokens=True)
    decoded = [normalize(decode) for decode in decoded]
    og_dataset_test_agen += decoded

og_dataset_test = og_dataset_test.drop(['prompt'], axis=1)
og_dataset_test['prediction-llama'] = og_dataset_test_agen
og_dataset_test.to_csv(DESTINATION_LLAMA_AGEN, index=False)

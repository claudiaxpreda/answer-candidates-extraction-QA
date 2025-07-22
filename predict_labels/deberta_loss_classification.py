
import sys 
import pandas as pd
import ast
import time
import prepare_input_classification as pic
import json
import os

import tensorflow as tf
import wandb
from wandb.integration.keras import WandbMetricsLogger
from sklearn.utils import shuffle


from transformers import AutoTokenizer, TFDebertaV2Model, TFBertModel
from tensorflow import keras
from tensorflow.keras import layers
from huggingface_hub import push_to_hub_keras

from sklearn.preprocessing import MultiLabelBinarizer


MAX_LEN_SEQ = 512
MAX_LEN_BERT = 768
HUGGING_TOKEN = ''
API_KEY_WB = ''
LR = 1e-4
LABELS_NUM = 4

def read_slice(source_file): 
  slice_df = pd.read_csv(source_file, keep_default_na=False, 
                           converters={"token_indexes": ast.literal_eval})
    
  return slice_df


def create_model(encoder_model_name):
  if encoder_model_name == 'google-bert/bert-base-uncased':
    encoder = TFBertModel.from_pretrained(encoder_model_name, output_hidden_states=True)
  else: 
    encoder =  TFDebertaV2Model.from_pretrained(encoder_model_name, output_hidden_states=True)



  context_ids = layers.Input(shape=(MAX_LEN_SEQ,), dtype=tf.int32)
  attention_mask = layers.Input(shape=(MAX_LEN_SEQ,), dtype=tf.int32)
  questions_ids = layers.Input(shape=(None, MAX_LEN_SEQ), dtype=tf.float32)
  embedding = encoder.deberta(
      context_ids, attention_mask=attention_mask
  )[0]

  #embedding: B, 512, 768
  #question_ids: B, MAX_LEN_PAD, 512
  questions_ids_t = tf.transpose(questions_ids, perm=[0, 2, 1])

  product_step_a=tf.matmul(questions_ids, embedding)
  
  product_step_b = tf.math.count_nonzero(questions_ids, 2, keepdims=True,  dtype=tf.dtypes.float32)

  product_step_c = tf.math.divide(product_step_a, tf.maximum(product_step_b, 1))

  layer1 = layers.TimeDistributed(
    layers.Dense(256, name="DenseLayer1", activation="relu"))
    (product_step_c, mask=tf.squeeze(product_step_b > 0, axis=-1))

  layer2 = layers.Dense(128, name="DenseLayer2")(layer1)

  layer3 = layers.Dense(LABELS_NUM, name="DenseLayer3",
            activation = "softmax")(layer2)

  output = layer5

  model = keras.Model(
      inputs=[context_ids, attention_mask, questions_ids],
      outputs= [output],
  )

  optimizer = keras.optimizers.Adam(learning_rate=LR)
  model.compile(
    optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

  return model


 
def main(answer_selection_model, max_len_pad, epochs, encoder_model_name):
  print('Program start time:'+ time.strftime("%H:%M:%S", time.localtime()))

  train_dataset_path = ''
  validation_dataset_path =  ''
  test_dataset_path =  ''

  tokenizer=AutoTokenizer.from_pretrained(encoder_model_name)
  mlb  = MultiLabelBinarizer(classes=[1, 2, 3, 4])

  input_train_df = shuffle(read_slice(train_dataset_path))
  input_validation_df = shuffle(read_slice(validation_dataset_path))

  x_train, y_train, _ = pic.prepare_input(input_train_df, tokenizer, int(max_len_pad), mlb)
  x_val, y_val, _ = pic.prepare_input(input_validation_df, tokenizer, int(max_len_pad), mlb)
  

  model = create_model(encoder_model_name)
  wandb.login(key = API_KEY_WB)

  configs_wb = {
      "learning_rate": LR,
      "architecture": encoder_model_name,
      "dataset": "FairytaleQALoss",
      "epochs": epochs,
    }
    
  run = wandb.init(
    project = "answer-selection",
    config = configs_wb, 
    notes = answer_selection_model,
  )

  model.fit(
    x_train,
    y_train,
    epochs= int(epochs),
    verbose=2,
    batch_size=48, 
    validation_data=(x_val, y_val),
    callbacks = [WandbMetricsLogger()]
  )

  run.finish()
  
  push_to_hub_keras(model, answer_selection_model, token = HUGGING_TOKEN)
  tokenizer.push_to_hub(answer_selection_model, token=HUGGING_TOKEN)
  
  print('Program end time:'+ time.strftime("%H:%M:%S", time.localtime()))

  return 0

if __name__ == "__main__":
  main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

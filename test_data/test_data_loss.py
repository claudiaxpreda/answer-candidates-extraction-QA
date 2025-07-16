import ast
import pandas as pd
import numpy as np
import sys
import tensorflow as tf
import transformers

from tensorflow import keras
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import confusion_matrix

from transformers import AutoTokenizer, AutoModelForCausalLM, TFDebertaV2Model
from huggingface_hub import push_to_hub_keras, from_pretrained_keras

sys.path.insert(0, '../predict_labels')
import prepare_input_classification as pic

sys.path.insert(1, '../predict_score')
import prepare_input as pi 

HUGGING_TOKEN = ''
PATH_TEST_LABELED = ''
DEBERTA_CLASSIFIER_NAME = ''
TOKENIZE_NAME = 'microsoft/deberta-v3-base'
DESTINATION_CLASSIFICATION_TEST = ''

test_dataset = pd.read_csv(PATH_TEST_LABELED, keep_default_na=False, 
                            converters={"token_indexes": ast.literal_eval})

# 1 - very good, 2 - good, 3 - average, 4 - unusable                   
mlb = mlb = MultiLabelBinarizer(classes=[1, 2, 3, 4])
model = from_pretrained_keras(DEBERTA_CLASSIFIER_NAME, token=HUGGING_TOKEN, compile=False)
tokenizer=AutoTokenizer.from_pretrained(TOKENIZE_NAME, token=HUGGING_TOKEN)

x_test, y_test, _ = pic.prepare_input(test_dataset, tokenizer, 100, mlb)

x = test_predictions.shape[0]
y = test_predictions.shape[1]

test_predictions = tf.reshape(test_predictions, (x*y, 4))
y_test = tf.reshape(y_test, (x*y, 4)).numpy()
mask = np.any(y_test, axis=1)

prob_score = test_predictions[:, 0]
prob_score = prob_score[mask]

Y = tf.one_hot(
    tf.math.argmax(test_predictions, axis = 1), depth = 4,  dtype=tf.int32)

Y_array = Y.numpy()
Y_array = Y_array[mask]

predicted_label = mlb.inverse_transform(Y_array)
predicted_label = [x for (x, ) in predicted_label]

gt_label = test_dataset['score_label'].to_list()

test_dataset['label_pred'] = predicted_label
test_dataset['prob_score'] = prob_score
test_dataset.to_csv(DESTINATION_CLASSIFICATION_TEST, index=False)
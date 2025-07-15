import os
import sys 
import pandas as pd
import ast
import numpy as np

from matplotlib import pyplot as plt


def read_slice(source_file): 
  slice_df = pd.read_csv(source_file, keep_default_na=False, 
                          converters={"token_indexes": ast.literal_eval})
  return slice_df

def write_slice(path, slice, df): 
  destination_file = path + 'input_' + slice + '.csv'
  df.to_csv(destination_file, index=False)


def concat_files(list_of_files, path, write_output=False, target_name=''):
  df_list = []

  for file in list_of_files: 
    df = pd.read_csv(path + file, index_col=None, header=0)
    df_list.append(df)
    print(len(df_list))
  
  df_concat = pd.concat(df_list, ignore_index=True)
  df_concat.drop_duplicates(inplace=True)

  if write_output == True: 
    df_concat.to_csv(target_name, index=False)
  
  return df_concat


def get_avg_len_pad():
  test_dataset_path = ''
  val_dataset_path = ''
  train_dataset_path = 'd'
  tokenizer=BertTokenizerFast.from_pretrained("bert-base-uncased")  
  max_len_pad = '100'
  
  x_train, y_train, gt = pi.prepare_input(read_slice(train_dataset_path), tokenizer, int(max_len_pad))
  print("Checkpoint 1: Generated data for trainig\n")

  x_val, y_val, gv = pi.prepare_input(read_slice(val_dataset_path), tokenizer, int(max_len_pad))
  print("Checkpoint 2: Generated data for val\n")

  x_test, y_test, gtt= pi.prepare_input(read_slice(test_dataset_path), tokenizer, int(max_len_pad))
  print("Checkpoint 3: Generated data for test\n")

  print("Train: " + str(gt.count().mean()['token_indexes']))
  print("\nVal: " + str(gv.count().mean()['token_indexes']))
  print("\nTest: " + str(gv.count().mean()['token_indexes']))


  print(gt.ngroups)
  print(gv.ngroups)
  print(gtt.ngroups)


SPLIT = 'test'

INDEXES_TRAIN = [
  '50000', '100000', '150000', '200000', 
  '250000', '300000', '350000', '400000',
  '500000', '600000', '700000', '800000', 
  '900000', 'end']

INDEXES_TEST_VAL = ['60000', 'end']

TASK_1 = 'concat'
TASK_2 = 'plot'
FOLDER = 'fairytale_dataset/set1/qloss/'
TARGET = f'fairytale_dataset/set1/final/{SPLIT}_all.csv'

task = 'concat'

if task == TASK_1:
  indexes = INDEXES_TEST_VAL
  list_of_files = []

  for index in indexes:
    list_of_files.append(f'ft_qloss_{SPLIT}_{index}.csv')

  print(len(list_of_files))
  df_concat = concat_files(list_of_files, FOLDER, write_output=True, target_name=TARGET)
  ax = df_concat.plot.hist(column=["rloss"], figsize=(10, 8), xticks = np.arange(0, 60, 5), yticks=np.arange(0,550000, 50000), xlabel = 'loss', ylabel = 'count')
  fig = ax.get_figure()
  fig.savefig(f'{SPLIT}_hist.png')
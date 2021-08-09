import pandas as pd
import glob
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import random

RAND_SEED = 42

np.random.seed(RAND_SEED)
torch.manual_seed(RAND_SEED)
random.seed(RAND_SEED)

def _get_train_val_test_size(dataset_len, split):
  assert sum(split) == 1

  train_size = round(dataset_len * split[0])
  val_size = round(dataset_len * split[1])
  test_size = dataset_len - train_size - val_size

  return train_size, val_size, test_size

def _get_all_files(dataset_dir):
  ftr_files_ls = glob.glob(f"{dataset_dir}/data_*.ftr")
  print(f"no of .ftr files: {len(ftr_files_ls)}")

  random.shuffle(ftr_files_ls)

  return ftr_files_ls

def _get_row_ids(dataset_dir):
  ftr_files_ls = glob.glob(f"{dataset_dir}/data_*.ftr")
  print(f"no of .ftr files: {len(ftr_files_ls)}")

  max_len = 0
  for file_path in ftr_files_ls:
    max_len = max(max_len, len(file_path))
  
  max_file_ls = []
  for file_path in ftr_files_ls:
    if len(file_path) == max_len:
      max_file_ls.append(file_path)
    elif len(file_path) > max_len:
      raise ValueError(f"max_len value {max_len} is not the maximum!")
  
  max_file_ls.sort()
  last_file = max_file_ls[-1]

  all_rows = []
  for file_path in tqdm(ftr_files_ls):
    if file_path == last_file:
      df = pd.read_feather(file_path)
      all_rows.extend([(file_path, i) for i in range(len(df))])
    else:
      all_rows.extend([(file_path, i) for i in range(100)])

  return all_rows

def _random_sample(row_ids:list, sample:int=None):
  random.shuffle(row_ids)
  if sample:
    if sample >= len(row_ids):
      raise ValueError(f"size of sample ({sample}) must be less than total no of rows ({len(row_ids)})")
    row_ids = row_ids[:sample]
  return row_ids

def _train_val_test_split(row_ids, split:tuple=(0.8, 0.1, 0.1)):
  random.shuffle(row_ids)

  train_size, val_size, test_size = _get_train_val_test_size(len(row_ids), split)

  train_set = row_ids[:train_size]
  val_set = row_ids[train_size:train_size+val_size]
  test_set = row_ids[train_size+val_size:train_size+val_size+test_size]

  return train_set, val_set, test_set

if __name__ == "__main__":
  dataset_dir = "dataset"
  sample_size = 20000 //100
  split = (.8, .1, .1)

  all_ftr_file_ls = _get_all_files(dataset_dir)
  # ds_sample = _random_sample(all_ftr_file_ls, sample=sample_size)
  train_set, val_set, test_set = _train_val_test_split(all_ftr_file_ls, split=split)

  print("saving sets to json...")
  with open("train_files.json", "w") as json_file:
    json.dump(train_set, json_file)
  with open("val_files.json", "w") as json_file:
    json.dump(val_set, json_file)
  with open("test_files.json", "w") as json_file:
    json.dump(test_set, json_file)


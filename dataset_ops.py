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

class TweetsCOV19Dataset(Dataset):
  def __init__(self, ds_rows):
    self.ds_rows = ds_rows
  
  def __len__(self):
    return len(self.ds_rows)

  def __getitem__(self, index):
    row_id = self.ds_rows[index]

    df = pd.read_feather(row_id[0])
    with open("output.txt", "w") as f:
      f.write(str(df.iloc[0].values))
    
    y = df["Retweets"][row_id[1]]
    x = df.drop(labels="Retweets", axis="columns").iloc[row_id[1]].values
    
    return np.array(x, dtype=float), np.array(y, dtype=int)

def _get_train_val_test_size(dataset_len, split):
  assert sum(split) == 1

  train_size = round(dataset_len * split[0])
  val_size = round(dataset_len * split[1])
  test_size = dataset_len - train_size - val_size

  return train_size, val_size, test_size

def get_dataset_generators(train_file, val_file, test_file, batch_size=64):
  print("loading datasets from files")
  with open(train_file) as json_file:
    train_set = json.load(json_file)
  train_set = TweetsCOV19Dataset(train_set)
  with open(val_file) as json_file:
    val_set = json.load(json_file)
  val_set = TweetsCOV19Dataset(val_set)
  with open(test_file) as json_file:
    test_set = json.load(json_file)
  test_set = TweetsCOV19Dataset(test_set)
  
  print("preparing dataloaders...")
  train_gen = DataLoader(train_set, shuffle=True, batch_size=batch_size)
  val_gen = DataLoader(val_set, shuffle=True, batch_size=batch_size)
  test_gen = DataLoader(test_set, shuffle=True, batch_size=batch_size)
  
  print(f"len(train_gen): {len(train_gen)}")
  print(f"len(val_gen): {len(val_gen)}")
  print(f"len(test_gen): {len(test_gen)}")

  return train_gen, val_gen, test_gen

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
  # # save all row_ids
  # dataset_dir = "dataset"
  # print("getting all row ids...")
  # row_ids = _get_row_ids(dataset_dir)
  # print("saving all row ids...")
  # with open("row_ids.json", "w") as json_file:
  #   json.dump(row_ids, json_file)
  
  # # load row_ids from file
  # print("loading all row ids from file...")
  # with open("row_ids.json") as json_file:
  #   row_ids = json.load(json_file)
  # print(f"len(row_ids): {len(row_ids)}")

  # sample_size = 200000
  # split = (0.7, 0.15, 0.15)
  batch_size = 32

  # print("sampling and splitting dataset...")
  # rand_row_ids = _random_sample(row_ids, sample=sample_size)
  # train_set, val_set, test_set = _train_val_test_split(rand_row_ids, split=split)

  # print("saving sets to json...")
  # with open("train.json", "w") as json_file:
  #   json.dump(train_set, json_file)
  # with open("val.json", "w") as json_file:
  #   json.dump(val_set, json_file)
  # with open("test.json", "w") as json_file:
  #   json.dump(test_set, json_file)

  # print("preparing dataloaders...")
  # train_gen = DataLoader(TweetsCOV19Dataset(train_set), shuffle=True, batch_size=batch_size)
  # val_gen = DataLoader(TweetsCOV19Dataset(val_set), shuffle=True, batch_size=batch_size)
  # test_gen = DataLoader(TweetsCOV19Dataset(test_set), shuffle=True, batch_size=batch_size)
  
  # print(f"len(train_gen): {len(train_gen)}")
  # print(f"len(val_gen): {len(val_gen)}")
  # print(f"len(test_gen): {len(test_gen)}")

  train_gen, val_gen, test_gen = get_dataset_generators("train.json", "val.json", "test.json", batch_size=batch_size)

  for x, y in train_gen:
    print(x)
    print(y)
    break

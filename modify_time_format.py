import pandas as pd
import glob
from tqdm import tqdm
import numpy as np

def split_timestamp_to_cols(df):
  total_seconds_in_day = 60 * 60 * 24
  total_months_in_year = 12

  timestamps = df.Timestamp
  second_of_day = pd.to_datetime(timestamps).dt.second + \
                  pd.to_datetime(timestamps).dt.minute * 60 + \
                  pd.to_datetime(timestamps).dt.hour * 60 * 60
  
  df["sin_second"] = np.sin(2*np.pi*second_of_day/total_seconds_in_day)
  df["cos_second"] = np.cos(2*np.pi*second_of_day/total_seconds_in_day)

  month_of_year = pd.to_datetime(timestamps).dt.month

  df["sin_month"] = np.sin(2*np.pi*month_of_year/total_months_in_year)
  df["cos_month"] = np.cos(2*np.pi*month_of_year/total_months_in_year)
  
  df["year"] = pd.to_datetime(timestamps).dt.year

  df.drop(labels="Timestamp", axis="columns", inplace=True)

  return df

def replace_files_with_correct_time(proj_root_path, dataset_dir):
  ftr_files_ls = glob.glob(f"{proj_root_path}/{dataset_dir}/data_*.ftr")
  for filename in tqdm(ftr_files_ls):
    df = pd.read_feather(filename)
    if "Timestamp" not in df.columns:
      continue
    df = split_timestamp_to_cols(df)
    df.to_feather(filename)

if __name__ == "__main__":
  proj_root_path = "/home/headquarters/Desktop/AI/project"
  dataset_dir = "dataset"
  replace_files_with_correct_time(proj_root_path, dataset_dir)
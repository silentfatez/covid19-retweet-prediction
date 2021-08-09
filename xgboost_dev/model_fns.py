import json
import os
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_log_error
from tqdm import tqdm
from xgboost.core import XGBoostError


class EarlyStopping:
  """Keep track of loss values and determine whether to stop model."""
  def __init__(self, 
               patience:int=10, 
               delta:int=0, 
               logger=None) -> None:
    self.patience = patience
    self.delta = delta
    self.logger = logger

    self.best_score = float("inf")
    self.overfit_count = 0

  def stop(self, loss):
    """Update stored values based on new loss and return whether to stop model."""
    threshold = self.best_score + self.delta

    # check if new loss is mode than threshold
    if loss > threshold:
      # increase overfit count and print message
      self.overfit_count += 1
      print_msg = f"Increment early stopper to {self.overfit_count} because val loss ({loss}) is greater than threshold ({threshold})"
      if self.logger:
        self.logger.info(print_msg)
      else:
        print(print_msg)
    else:
      # reset overfit count
      self.overfit_count = 0
    
    # update best_score if new loss is lower
    self.best_score = min(self.best_score, loss)
    
    # check if overfit_count is more than patience set, return value accordingly
    if self.overfit_count >= self.patience:
      return True
    else:
      return False

def train(train_set:list, 
          val_set:list, 
          n_epochs:int, 
          relevant_features:list,
          epoch_set_size:int,
          train_val_ratio:tuple,
          num_boost_round:int=10,
          booster:str="gbtree",
          load_checkpoints:str=None, 
          save_prefix:str=None, 
          es_patience:int=10):
  
  params = {
      "tree_method": "approx",
      "booster": booster
  }
  model = None
  stats = []
  load_prefix = load_suffix = None
  early_stopper = EarlyStopping(patience=es_patience)

  if load_checkpoints:
    load_prefix, load_suffix = load_checkpoints.split("*")
    save_prefix = load_prefix
  else:
    if not save_prefix:
      raise ValueError("if load_checkpoints are not identified, please include save_prefix")
  
  for epoch_no in range(1, n_epochs+1):
    epoch_train_ls = random.choices(train_set, k=epoch_set_size*train_val_ratio[0])
    epoch_val_ls = random.choices(val_set, k=epoch_set_size*train_val_ratio[1])
    print(f"Epoch {epoch_no}:")

    if epoch_no == 2:
      params.update({'process_type': 'update',
                    'updater'     : 'refresh',
                    'refresh_leaf': True,
                    'verbosity': 0})
    
    loaded = False
    if load_prefix and load_suffix:
      load_chkpt = f"{load_prefix}{epoch_no}{load_suffix}"
      if os.path.isfile(load_chkpt):
        model=xgb.Booster()
        model.load_model(load_chkpt)
        params.update({'process_type': 'update',
                        'updater'     : 'refresh',
                        'refresh_leaf': True,
                        'verbosity': 0})
        loaded = True

    if not loaded:
      try:
        model = _train_epoch(epoch_train_ls, model, params, num_boost_round, relevant_features)
        model.save_model(f"{save_prefix}{epoch_no}.model")
      except XGBoostError as xgb_err:
        print(f"XGBoostError thrown: {xgb_err}")
        print("Returning current model and stats...")
        return model, stats
    
    overall_msle = _val_epoch(epoch_val_ls, model, relevant_features)
    print(f"MSLE: {overall_msle}")
    stats.append(overall_msle)
    
    # update and check early stopper
    if early_stopper.stop(overall_msle):
      print("Model has overfit, early stopping...")
      break
    
  return model, stats

def load_model(checkpt_path):
  model=xgb.Booster()
  model.load_model(checkpt_path)
  return model

def predict(model:xgb.Booster, 
            x:np.ndarray, 
            feature_names:list=None):
  data_matrix = xgb.DMatrix(x, feature_names=feature_names)
  y_pred = model.predict(data_matrix)
  y_pred = y_pred.clip(min=0)
  y_pred = np.nan_to_num(y_pred)
  return y_pred

def _train_epoch(train_file_ls, 
                model,
                params, 
                num_boost_round, 
                relevant_features):
  for train_file in tqdm(train_file_ls):
    train_df = pd.read_feather(train_file)
    x, y, feature_names = _extract_x_y_featurenames(train_df, 
                                                    relevant_features)
    dtrain = xgb.DMatrix(x, 
                          label=y, 
                          feature_names=feature_names)
    model = xgb.train(params, 
                      dtrain, 
                      num_boost_round=num_boost_round, 
                      xgb_model=model)
  return model

def _val_epoch(val_file_ls, 
               model,
               relevant_features):
  overall_msle = 0
  for val_file in tqdm(val_file_ls):
    val_df = pd.read_feather(val_file)
    x, y, feature_names = _extract_x_y_featurenames(val_df, 
                                                    relevant_features)
    y_pred = predict(model, x, feature_names)
    msle = mean_squared_log_error(y, y_pred)
    overall_msle += msle
  overall_msle /= len(val_file_ls)
  return overall_msle

def _extract_x_y_featurenames(df, 
                              relevant_features):
  y = df["Retweets"].values
  x = df[relevant_features]
  feature_names = x.columns
  x = x.values
  y = np.nan_to_num(y)
  return x, y, feature_names

if __name__ == "__main__":
  model = load_model("model1b_ep10.model") # replace model file path
  df = pd.read_feather("dataset/data_0.ftr") # replace data file path
  with open("relevant_features.json") as f:
    relevant_features = json.load(f)
  x, y, feature_names = _extract_x_y_featurenames(df, 
                                                  relevant_features)
  x = x[1] # note index of row which we are interested in
  print(y)
  x = np.reshape(x,(1, x.size))
  prediction = predict(model, x)
  print(prediction)

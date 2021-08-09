# XGBoost

This directory contains code for the use of XGBoost algorithm to tackle the challenge.

## Directory Structure

```utf-8
.
├── imgs/                   # images generated to present statistics
├── saved_models/           # saved models from experimentation
├── model_fns.py            # helper functions to train model
├── run_experiments.ipynb   # notebook with experimental process
├── visualisations.ipynb    # notebook with code to generate visualisations
├── requirements.txt        # requirements for code to work
├── relevant_features.json  # json file with identified relevant features
├── stats_train.json        # json file with train statistics
├── stats_val.json          # json file with validation statistics
└── README.md
```

## Instructions

1. Write a short script to change the paths to the different files in the train/val/test sets to load the file from where the dataset is saved accordingly. In the code here, it is assumed that there is the dataset is contained in a `dataset/` directory in this directory.
2. Run the `run_experiments.ipynb` notebook in sequential order.

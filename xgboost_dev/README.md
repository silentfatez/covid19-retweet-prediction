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

Run the `run_experiments.ipynb` notebook in sequential order.

## Summary of Experiments

### Experiment 1 - Varying Number of Estimators

| Model Name   | No. of Estimators | MSLE               |
|------------- | ----------------- | ------------------ |
| model1a_ep23 | 2                 | 0.4214346885158393 |
| model1b_ep9  | 4                 | 0.3680438705400346 |
| model1c_ep7  | 6                 | 0.3758612621173864 |
| model1d_ep7  | 8                 | 0.3667101573014846 |
| model1e_ep3  | 10                | 0.35703752802547395 |

### Experiment 2 - Presence of Dropout

In the interest of time, the number of estimators used in this experiment is 2.

| Model Name   | Presence of Dropout | MSLE               |
|------------- | ------------------- | ------------------ |
| model1a_ep23 | No                  | 0.4214346885158393 |
| model2_ep3   | Yes                 | 0.4922658251014944 |

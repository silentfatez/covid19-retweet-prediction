# COVID-19 Retweet Prediction

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

During the Covid-19 Pandemic, we have seen the spread of fake news causing life-threatening situations. Case in point, due to the misinformation spread by the United State’s former President Trump on chloroquine, an Arizona man consumed the chemical believing it would cure Covid-19, and subsequently passed on. 

To prevent these situations from happening in the future, screening of social media content for fake news so as to remove them before they spread is one of the ways which can be employed to tackle this issue. Due to limited time and resources, predicting the most viral content for screening would greatly aid in efforts to combat fake news. In the case of Twitter, virality can be measured by the number of retweets.

Thus, this project aims to predict the number of retweets based on other features collected on the tweet. During the course of this project, 3 types of models were explored, namely:

1. eXtreme Gradient Boosting (XGBoost)
2. Linear Neural Networks
3. Long Short-Term Memory (LSTM)

The dataset used in this project is the [TweetsCOV19 Dataset](https://data.gesis.org/tweetscov19/), which is a sementically annotated corpus of Tweets about the COVID-19 pandemic. It consists of 20,112,480 tweets in total, posted by 7,384,417 users and reflects the societal discourse about COVID-19 on Twitter in the period of October 2019 until December 2020.

## Directory Structure

```utf-8
.
├── dataset/                          # pre-processed dataset
├── linear_neural_network/            # code for linear neural networks
├── lstm/                             # code for lstm models
├── xgboost/                          # code for xgboost algorithm
├── combine_df.py                     # scripts to combine dataset parts
├── dataset_ops.py                    # helper functions split data
├── data_exploration.ipynb            # notebook for data exploration
├── parallel_compute_clean_data.ipynb # notebook for cleaning data
├── train_files_801010.json           # json with train set
├── val_files_801010.json             # json with validation set
├── test_files_801010.json            # json with test set
└── README.md
```

## Model Development

1. Download [pre-processed dataset](https://drive.google.com/file/d/1ZVpFhf0iZ_BfJWP4kVYgvLMPf_Yb039t/view?usp=sharing) and place it in the `./dataset/` directory.
2. Navigate to the directory with the model of interest. E.g. if you want to check out the xgboost model, enter `cd xgboost` in the terminal.
3. Create a virtual environment.

    ```bash
    virtualenv venv
    ```

4. Activate the virtual environment.

    ```bash
    source venv/bin/activate
    ```

5. Install required packages.

    ```bash
    pip install -r requirements.txt
    ```

6. Follow the instructions in the READMEs of the respective model directories (i.e. `linear_neural_network/`, `lstm/` and `xgboost/`) to run load, build and/or test the models.

## Demonstration

A demonstration of the models can be seen on [this site](https://covid19-retweet.as.r.appspot.com).

For instructions to how to run the GUI locally, go to [this repository](https://github.com/cre8tion/COVID19-Retweet-UI).

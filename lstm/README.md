# LSTM

This directory contains code for the use of LSTM to tackle the challenge.

## Directory Structure

```utf-8
.
├── lstm_model_maker.ipynb  # notebook with code to make models
├── lstm_tester.ipynb       # notebook with code to test model
└── README.md
```

## Instructions

Run the `lstm_model_maker.ipynb` notebook in sequential order to build models.

Run the `lstm_tester.ipynb` notebook in sequential order to test models.

## Hyperparameters

### Model 1
model parameters: ([256], 30,2, 0, True,0.5)

* Hidden dimension: 256
* sequence length: 30
* LSTM hidden layers: 2
* Linear Neural Network Layers: 0
* Bidirectional: True
* dropout: 0.5

### Model 2

model parameters: ([256,256], 30,2, 1, True,0.5)

* Hidden dimension: 256(LSTM), 256(Linear)
* sequence length: 30
* LSTM hidden layers:2
* Linear Neural Network Layers: 1
* Bidirectional: True
* dropout: 0.5

### Model 3

model parameters: ([512,512], 30,2, 1, True,0.5)

* Hidden dimension: 512(LSTM), 512(Linear)
* sequence length: 30
* LSTM hidden layers: 2
* Linear Neural Network Layers: 1
* Bidirectional: True
* dropout: 0.5

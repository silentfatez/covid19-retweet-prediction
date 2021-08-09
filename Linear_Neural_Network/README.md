# Linear Neural Network

This directory contains code for the use of Linear Neural Network to tackle the challenge.

## Directory Structure

```utf-8
.
├── templates/
    ├── model_maker
    ├── model_validation_maker
    ├── model_test
├── model_trained/
    ├── model_1/
        ├── model_1.ipynb
        ├── model_1_validation.ipynb
        ├── state_dict_1.ipynb
    ├── model_2/
        ├── model_2.ipynb
        ├── model_2_validation.ipynb
        ├── state_dict_2.ipynb
        ├── state_dict_after_all_epochs_2.ipynb
    ├── model_3/
        ├── model_3.ipynb
        ├── model_3_validation.ipynb
        ├── state_dict_3.ipynb
        ├── state_dict_after_all_epochs_3.ipynb
    ├── best_model_test.ipynb
├── predict.ipynb
├── requirements.txt
├── README.md
```
## Explanation

All the models have 2 hidden layers with a dimension of 256 each before finally projected out a singular result (retweets)

### Templates

The templates for the model are stored in

```
    cd ./templates
```

### Model 1

**HyperParameters**
* Learning Rate: 0.1
* Step size: 50
* Dropout Rate: 0.5

The model training can be found in
```
    cd ./models_trained
    cd ./model_1
    model_1.ipynb
```
The validation perfomance of this model can be found in
``` 
    cd ./models_trained
    cd ./model_1
    model_1_validation.ipynb
```
The model is saved in
```
    cd ./models_trained
    cd ./model_1
    state_dict_1.ipynb
```
### Model 2

**HyperParameters**
* Learning Rate: 0.001
* Step size: 10
* Dropout Rate: 0.6

The model training can be found in
```
    cd ./models_trained
    cd ./model_2
    model_2.ipynb
```
The validation perfomance of this model can be found in
```
    cd ./models_trained
    cd ./model_2
    model_2_validation.ipynb
```
The model is saved in
```
    cd ./models_trained
    cd ./model_2
    state_dict_2.ipynb
```
### Model 3

**HyperParameters**
* Learning Rate: 0.001
* Step size: 25
* Dropout Rate: 0.5

The model training can be found in
```
    cd ./models_trained
    cd ./model_3
    model_3.ipynb
```
The validation perfomance of this model can be found in
```
    cd ./models_trained
    cd ./model_3
    model_3_validation.ipynb
```
The model is saved in
```
    cd ./models_trained
    cd ./model_3
    state_dict_3.ipynb
```

### Best Model
The best model performance on the test set can be found in
```
    cd ./models_trained
    best_model_test.ipynb
```

### Prediction
To predict a result
```
    predict.ipynb
```

# Driver Safety Challenge for Grab

This repository contains the code for classifying vehicle trips based on the telematics data.
The code has been developed and tested on a machine running latest version of Windows 10.

## Data
https://www.aiforsea.com/safety

The data consists of features; acceleration readings, gyro readings and speed etc and a binary target label for dangerous driving

## Requirements
* Keras==2.2.4
* numpy==1.16.2
* imbalanced_learn==0.4.3
* pandas==0.24.2
* imblearn==0.0
* scikit_learn==0.21.2

## Workflow
1. TRAINING
      
    - The training dataset is split into training and testing sets. Standard scaling is applied to transform the data. Stratified KFolds with k = 10 is then utilised to cross validate the dense sequential model which makes use of a combination of layers with relu as activation function. Final layer layer uses sigmoid activation to output class probability. Early stopping has been added to stop the training once validation loss stop improving. The best model from each fold is saved on disk. For each fold model is evaluated on the initially split test set. 
    - in order to recreate the model, follow the below instructions:
      1. After cloning the repository, place the training data and corresponding label files in the \\data folder.
      2. Run main.py providing train for the -m argument and the \\path\\to\\clonedrepository for -d argument. Example as follows
          > python main.py -m train -d C:\Users\DELL\Downloads\AIforSEA\safety
2. TESTING

    - All data provided for testing is first transformed and then scaled using the same scaling values from the training data after which the it is run through the best model achieved from the KFolds cross validation.
    - to run the test code, follow the below instructions:
      1. After cloning the repository, place testing data and corresponding label files in the data\\holdout folder.
      2. Run main.py providing test for the -m argument and the \\path\\to\\clonedrepository for -d argument. Example as follows
          > python main.py -m test -d C:\Users\DELL\Downloads\AIforSEA\safety

## Code Description
- main.py - helper functions are called here using train and test function depending on the arguments provided
- prep.py - functions used to transform and create features from raw data 
- training.py - functions for creating model structure, training and testing

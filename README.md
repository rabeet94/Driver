# Driver Safety Challenge for Grab

This repository contains the code for classifying vehicle trips based on the telematics data.
The code has been developed and tested on a machine running latest version of Windows 10.

1. Data: https://www.aiforsea.com/safety
    The data consists of features; acceleration readings, gyro readings and speed etc and a binary target label for dangerous driving.
    
2. Workflow
    REQUIREMENTS
      Before running the code, ensure following packages are installed in the environment.
    TRAIN
    - in order to recreate the model, follow the below instructions:
      1. After cloning the repository, place the training data and corresponding label files in the \\data folder.
      2. 
    TEST
    - to run the test code, follow the below instructions:
      1. After cloning the repository, place testing data and corresponding label files in the data\\holdout folder.
      2. Run main.py providing test for the -m argument and the \\path\to\clonedrepository for -d argument.
          > python main.py -m test -d C:\Users\DELL\Downloads\AIforSEA\safety

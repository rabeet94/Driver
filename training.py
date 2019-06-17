# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:39:11 2019

@author: DELL
"""

import pandas as pd
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, roc_auc_score

from keras.layers import Dense
from keras.layers import Dropout
from keras.models import Sequential
from keras.models import load_model

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from imblearn.over_sampling import SMOTE

def train_model(fframe):    
    X_train, X_test, y_train, y_test = train_test_split(fframe.drop(['label', 'maxseconds'], axis = 1),
                                                        fframe['label'],
                                                        test_size = 0.2, stratify = fframe['label'])
    
    sc = StandardScaler()
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns = X_train.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns = X_test.columns)
    joblib.dump(sc, 'standardscaler.bin')
    
    sm = SMOTE(sampling_strategy = 'minority', random_state=30)
    upx, upy = sm.fit_sample(X_train, y_train)
    X_train = pd.DataFrame(upx, columns = X_train.columns)
    y_train = pd.DataFrame(upy, columns = ['label'])
    
    del upx, upy, fframe
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    fold = 1
    for train, test in kfold.split(X_train, y_train):
        model = Sequential()
        model.add(Dense(1024, activation='relu', input_dim = 33))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        filepath = os.getcwd() + "\\models\\best_model_fold_%s.hdf5" % fold
        filepath2 = os.getcwd() + "\\models\\final_model_fold_%s.hdf5" % fold
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
        mc = ModelCheckpoint(filepath, monitor='val_acc', mode='max', verbose=1, save_best_only=True)
        
        model.fit(X_train.values[train], y_train.values[train],
                  epochs=3000, batch_size=512, verbose=False,
                  validation_data=(X_train.values[test], y_train.values[test]),
                  callbacks = [es, mc])
    
        model.save(filepath2)
        saved_model = load_model(filepath)
        _, train_acc = saved_model.evaluate(X_train.values[train], y_train.values[train], verbose=0)
        _, val_acc = saved_model.evaluate(X_train.values[test], y_train.values[test], verbose=0)
        _, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
        print('Best model scores')
        print('Train: %.3f, Val: %.3f, Test: %.3f' % (train_acc, val_acc, test_acc))
        fold+=1
    
def holdout_predictions(model_path, holdout_dataframe):
    sc = joblib.load('standardscaler.bin')
    holdout_dataframe = holdout_dataframe.drop(['label', 'maxseconds'], axis  = 1)
    holdout_dataframe = pd.DataFrame(sc.transform(holdout_dataframe), columns = holdout_dataframe.columns)
    
    model = load_model(model_path)
    predictions = model.predict(holdout_dataframe)
    predictions = (predictions > 0.5).astype(np.int)
    return predictions
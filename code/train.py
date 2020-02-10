#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 23:20:39 2020

@author: chienchichen
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from utils import Discriminator_Model
from sklearn.metrics import f1_score, accuracy_score

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
set_session(session)

class Train():
    def __init__(self, name,base_path, train_path,val_path,test_path, batch_size=50, epochs = 20):
        self.name = name
        self.batch_size = batch_size
        self.epochs = epochs
        self.history = None
        self.base_path = base_path
        self.df_train = pd.read_csv(train_path)
        self.df_val = pd.read_csv(val_path)
        self.df_test = pd.read_csv(test_path)
        self.xtrain = None
        self.ytrain = None
        self.xval = None
        self.yval = None
        self.xtest = None
        self.ytest = None
        
    def prepare_input(self):     
        self.xtrain, self.ytrain = self._read_data_df(self.df_train)
        self.xval, self.yval = self._read_data_df(self.df_val)
        self.xtest, self.ytest = self._read_data_df(self.df_test)
        
    def _read_data_df(self, df):
        X = []
        y = []
        for i in range(len(df)):
            path = os.path.join(self.base_path, df.iloc[i,2],df.iloc[i,1],df.iloc[i,0])
            X.append(np.load(path).T)
            y.append(1 if df.iloc[i,1]=="REAL" else 0)
        return X,y
        
 
measure_performance_only = False
if measure_performance_only:
    pretrained_model_name = 'saved_model_240_8_32_0.05_1_50_0_0.0001_500_156_2_True_True_fitted_objects.h5'
    model_class = Discriminator_Model(load_pretrained=True, saved_model_name=pretrained_model_name, real_test_mode=True)
else:
    pretrained_model_name = 'saved_model_240_8_32_0.05_1_50_0_0.0001_100_156_2_True_True_fitted_objects.h5'
    # model_class = Discriminator_Model(load_pretrained=True, saved_model_name=pretrained_model_name, real_test_mode=False)
    model_class = Discriminator_Model(load_pretrained=False, real_test_mode=False)
    
name, base_path, train_path,val_path,test_path, batch_size, epochs = 'audio','./../../../dataset/fb_audio_pre','./../audio_train.csv','./../audio_val.csv','./../audio_test.csv', 20, 20
train = Train(name, base_path, train_path,val_path,test_path, batch_size=batch_size, epochs=epochs)
train.prepare_input()
print("Initializing model")


print("training model")
model_class.train(train.xtrain, train.ytrain, train.xval, train.yval)

print("optimizing threshold probability")
model_class.optimize_threshold(train.xtrain, train.ytrain, train.xval, train.yval)
print(f"optimum threshold is {model_class.opt_threshold}")
print("evaluating trained model")
model_class.evaluate(train.xtrain, train.ytrain, train.xval, train.yval)
print("calculating test performance")
y_test_pred_labels = model_class.predict_labels(train.xtest, threshold=model_class.opt_threshold)
test_acc = accuracy_score(train.ytest, y_test_pred_labels)
test_f1_score = f1_score(train.ytest, y_test_pred_labels)
print(f"Test set accuracy: {test_acc}, f1_score: {test_f1_score} ")

print("calculating performance on real test set")
real_test_acc_score, real_test_f1_score_val = model_class.inference_on_real_data(threshold=model_class.opt_threshold)
print(f"Realtalk test set accuracy: {real_test_acc_score}, f1_score: {real_test_f1_score_val} ")

# try:
#     foundations.log_metric('test_accuracy', np.round(test_acc, 2))
#     foundations.log_metric('test_f1_score', np.round(test_f1_score, 2))
#     foundations.log_metric('realtalk_accuracy', np.round(real_test_acc_score, 2))
#     foundations.log_metric('realtalk_f1_score', np.round(real_test_f1_score_val, 2))

# except:
#     print("foundations command not found")

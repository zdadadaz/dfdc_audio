#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:49:05 2020

@author: chienchichen
"""
import numpy as np
import pandas as pd
import os
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# path_json = "./../../jcc_dfdc/playground/db_audio_fr.csv"
# df1 = pd.read_csv(path_json)
# real_num = sum((df['label']=="REAL") & (df['folder']=="dfdc_train_part_0"))
# fake_num = sum((df['label']=="FAKE") & (df['folder']=="dfdc_train_part_0"))

# folder_0_num = sum( (df['folder']=="dfdc_train_part_0"))

def read_pre_files():
    filename =[]
    label = []
    folders = []
    
    path_dir = "./../../dataset/fb_audio_pre"
    for folder in os.listdir(path_dir):
        if folder[0] == ".":
            continue
        for rf in os.listdir(os.path.join(path_dir, folder)):
            if rf[0] == ".":
                continue
            for f in os.listdir(os.path.join(path_dir, folder, rf)):
                if f[0] == ".":
                    continue
                filename.append(f)
                label.append(rf)
                folders.append(folder)
    
    d = {'filename': filename, 'label': label, 'folder': folders}
    df = pd.DataFrame(data=d)
    
    real_num = sum((df['label']=="REAL"))
    fake_num = sum((df['label']=="FAKE"))
    df.sort_values('folder').to_csv('audio_dataset_pre.csv', index=False)


def create_train_valid_test(csv_path):
    df = pd.read_csv(csv_path)
    real_df = df[df['label']=="REAL"]
    fake_df = df[df['label']=="FAKE"]
    
    X_train_tmp, X_test, y_train_tmp, y_test = train_test_split(real_df, real_df['label'], test_size=0.1, random_state=42)
    X_train_f_tmp, X_test_f, y_train_f_tmp, y_test_f = train_test_split(fake_df, fake_df['label'], test_size=0.1, random_state=42)
    
    X_train, X_val, y_train, y_val = train_test_split(X_train_tmp, y_train_tmp, test_size=0.2, random_state=42)
    X_train_f, X_val_f, y_train_f, y_val_f = train_test_split(X_train_f_tmp, y_train_f_tmp, test_size=0.2, random_state=42)
    
    X_train = X_train.append(X_train_f)
    X_val = X_val.append(X_val_f)
    X_test = X_test.append(X_test_f)
    
    X_train.to_csv('audio_train.csv',index = False)
    X_val.to_csv('audio_val.csv',index = False)
    X_test.to_csv('audio_test.csv',index = False)

    
    
    
csv_path = "audio_dataset_pre.csv"
csv_path_audio = "audio_dataset.csv"
fake_audio_path = "metadata_audio_altered.csv"
# create_train_valid_test(csv_path)
# df = pd.read_csv(csv_path)
# df_file = pd.read_csv(csv_path_audio)
# df_fa = pd.read_csv(fake_audio_path)
# seed = 100
# total_real = sum(df['label']=="REAL")
# total_fake = sum(df['label']=="FAKE")



# real_re = resample(df[df['label']=="REAL"], n_samples=int(total_real*0.9), replace=False, random_state=seed)
# fake_re = resample(df[df['label']=="FAKE"], n_samples=int(total_fake*0.9), replace=False, random_state=(seed+300))



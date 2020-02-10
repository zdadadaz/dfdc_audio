#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 17:14:17 2020

@author: chienchichen
"""

import os
import numpy as np
import subprocess
from sklearn.metrics import f1_score, accuracy_score
from utils import *
import matplotlib.pyplot as plt
import nlpaug.augmenter.audio as naa

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




class Run_model():
    def __init__(self, name, dir_path, out_dir_path):
        self.name = name
        self.dir_path = dir_path
        self.out_dir_path = out_dir_path
        
    def process_audio_files_inference(self, audio_path):
        audio_array, sample_rate = librosa.load(audio_path, sr=16000)
        # trim_audio_array, index = librosa.effects.trim(audio_array)
        mel_spec_array = melspectrogram(audio_array, hparams=hparams)
        return mel_spec_array

    def process_audio_files_with_aug(self, audio_path):
        sr = 16000
        audio_array, sample_rate = librosa.load(audio_path, sr=sr)
        aug_crop = naa.CropAug(sampling_rate=sr)
        audio_array_crop = aug_crop.augment(audio_array)
        aug_loud = naa.LoudnessAug(loudness_factor=(2, 5))
        audio_array_loud = aug_loud.augment(audio_array)
        aug_noise = naa.NoiseAug(noise_factor=0.03)
        audio_array_noise = aug_noise.augment(audio_array)
    
        mel_spec_array_load = melspectrogram(audio_array_loud, hparams=hparams)
        mel_spec_array_noise = melspectrogram(audio_array_noise, hparams=hparams)
    
        audio_array_list= [mel_spec_array_load, mel_spec_array_noise ]
        
        # audio_array_list= [audio_array_crop,audio_array_loud,
        #                    audio_array_noise ]
    
        return audio_array_list

    def run(self):
        tmp = []
        # tmp = ["dfdc_train_part_9", "dfdc_train_part_1", "dfdc_train_part_8","dfdc_train_part_23",
        #        "dfdc_train_part_19","dfdc_train_part_16","dfdc_train_part_29","dfdc_train_part_14"]

        # tmp = ["dfdc_train_part_9","dfdc_train_part_1","dfdc_train_part_8","dfdc_train_part_23",\
        #         "dfdc_train_part_19","dfdc_train_part_16", "dfdc_train_part_29"]
        #  to 28
        for folder in os.listdir(self.dir_path):
            if folder[0] =="." or folder in tmp:
                continue
            print(folder)
            for df_real in os.listdir(os.path.join(self.dir_path, folder)):
                if df_real[0] == ".":
                    continue
                if df_real == 'FAKE':
                    continue
                for file in os.listdir(os.path.join(self.dir_path, folder,df_real)):
                    if  (file[-4:] != '.wav'):
                        continue
                    outpath = os.path.join(self.out_dir_path, folder, df_real)              
                    audio_path = os.path.join(self.dir_path, folder,df_real, file)
                    # coef_img = self.process_audio_files_inference(audio_path)
                    # self.write_coef(coef_img, outpath, file)
                    coef_imgs = self.process_audio_files_with_aug(audio_path)
                    clist = ['crop', 'noise']
                    for c in range(2):
                        coef_img = coef_imgs[c]
                        file_name = file[:-4] + "_"+ clist[c] + ".wav"
                        self.write_coef(coef_img, outpath, file_name)
                    
    def write_coef(self, coef_img, path, file):
        # np.savetxt(os.path.join(path,file[:-4]+".txt"), coef_img, delimiter=',')
        np.save(os.path.join(path,file[:-4]+".npy"), coef_img)

dir_path = "./../../../dataset/fb_audio"
out_dir_path = "./../../../dataset/fb_audio_pre"
run = Run_model("preprocess_audio", dir_path, out_dir_path)
run.run()

aa = load(os.path.join(out_dir_path,'dfdc_train_part_43', 'REAL','keddthzqkd.npy'))

# make directories
# for i in range(50):
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder)
#     cmd = 'mkdir ' + path
#     os.system(cmd)
    
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder,"REAL")
#     cmd = 'mkdir ' + path
#     os.system(cmd)
    
#     folder = "dfdc_train_part_"+str(i)
#     path = os.path.join(out_dir_path,folder,"FAKE")
#     cmd = 'mkdir ' + path
#     os.system(cmd)

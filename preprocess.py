# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import numpy as np
import scipy.signal as sig
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import os
import sys
import json
import pandas as pd
import wfdb
import ast
from scipy.stats import zscore
from ECG_deeplearning import ecg_noisecancellation_for_deeplearning
def PTB_labelconvert(label_str):
    label_set=[]
    for i in range(len(label_str)):
        one_hot=np.zeros((1,5))
        if 'NORM' in label_str[i]:
            one_hot[0,0]=1
        if 'MI' in label_str[i]:
            one_hot[0,1]=1
        if 'CD' in label_str[i]:
            one_hot[0,2]=1
        if 'STTC' in label_str[i]:
            one_hot[0,3]=1
        if 'HYP' in label_str[i]:
            one_hot[0,4]=1
        label_set.append(one_hot)
    label_set=np.vstack(label_set)
    return label_set

def normalize(X_train, X_validation,X_test, ntype='sample'):
    # Standardize data such that mean 0 and variance 1
    if ntype == 'sample':
        chanel,length=X_train.shape[1],X_train.shape[2]
        X_train=X_train.reshape((X_train.shape[0],chanel*length))
        X_test = X_test.reshape((X_test.shape[0], chanel * length))
        X_validation=X_validation.reshape((X_validation.shape[0],chanel*length))
        X_train=zscore(X_train)
        X_validation= zscore(X_validation)
        X_test = zscore(X_test)
        X_train=X_train.reshape((X_train.shape[0],chanel,length))
        X_validation = X_validation.reshape((X_validation.shape[0], chanel, length))
        X_test = X_test.reshape((X_test.shape[0], chanel, length))
    else:
        ss = StandardScaler()
        ss.fit(X_train.flatten()[:, np.newaxis].astype(float))
        X_train,X_validation=apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss)
    return X_train,X_validation,X_test
def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data
def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp
def preprocess_signal(x, preprocess_cfg, sample_rate,max_length=9000):
    """ resample, filter, scale, ecg signal """
    # if sample_rate != preprocess_cfg.sample_rate:
    #     num = x.shape[1] // sample_rate * preprocess_cfg.sample_rate
    #     x = sig.resample(x, int(num), axis=1)
    #     sample_rate = preprocess_cfg.sample_rate
    # print('before')
    # print(x.shape)
    x = filter_signal(x, preprocess_cfg, sample_rate)
    # num_leads=x.shape[0]
    # denoised_signal=np.zeros((num_leads,max_length))
#    for i in range(x.shape[0]):
#        denoised_signal[i,:] = ecg_noisecancellation_for_deeplearning(x[i,:].squeeze(), sample_rate,max_length)
#        if np.isnan(denoised_signal[i,:]).any()==True:
#            denoised_signal[i, :]=np.nan_to_num(denoised_signal[i, :],nan=np.nanmean(denoised_signal[i, :]))
    denoised_signal=x
    denoised_signal = scale_signal(denoised_signal, preprocess_cfg)
    return denoised_signal


def filter_signal(x, preprocess_cfg, sample_rate):
    """ filter ecg signal """
    nyq = sample_rate * 0.5
    for i in range(len(x)):
        for cutoff in preprocess_cfg.filter_highpass:
            x[i,:] = sig.filtfilt(*sig.butter(3, cutoff / nyq, btype='highpass'), x[i,:])
        for cutoff in preprocess_cfg.filter_lowpass:
            if cutoff >= nyq: cutoff = nyq - 0.05
            x[i,:] = sig.filtfilt(*sig.butter(3, cutoff / nyq, btype='lowpass'), x[i,:])
        if len(preprocess_cfg.filter_bandpass)>0:
            x[i,:] = sig.filtfilt(*sig.butter(3, [preprocess_cfg.filter_bandpass[0] / nyq, preprocess_cfg.filter_bandpass[1] / nyq], btype='bandpass'), x[i,:])
        for cutoff in preprocess_cfg.filter_notch:
            x[i,:] = sig.filtfilt(*sig.iirnotch(cutoff/nyq,30), x[i,:])
    return x


def scale_signal(x, preprocess_cfg):
    """ scale ecg signal """
    for i in range(len(x)):
        if preprocess_cfg.scaler is None: continue
        elif "minmax" in preprocess_cfg.scaler:   scaler = MinMaxScaler()
        elif "standard" in preprocess_cfg.scaler: scaler = StandardScaler()
        elif "robust" in preprocess_cfg.scaler:   scaler = RobustScaler()
        scaler.fit(np.expand_dims(x[i,:], 1))
        x[i,:] = scaler.transform(np.expand_dims(x[i,:], 1)).squeeze()

    return x


def augment_signal(x):
    """ augmentations (scale, noise) """
    for i in range(len(x)):
        scale = np.random.normal(loc=1.0, scale=0.1)
        noise = np.random.normal(loc=0.0, scale=0.1, size=x[i].shape)
        x[i,:] = x[i,:] * scale + noise

    return x


def preprocess_label(labels, scored_classes, equivalent_classes):
    """ convert string labels to binary labels """
    y = np.zeros((len(scored_classes)), np.float32)
    for label in labels:
        if label in equivalent_classes:
            label = equivalent_classes[label]

        if label in scored_classes:
            y[scored_classes.index(label)] = 1

    return y

class PreprocessConfig():
    def __init__(self, file=None, idx="preprocess_config"):
        """ preprocess configurations """
        self.idx = idx
        self.all_negative = None
        self.sample_rate = None
        self.filter_notch = []
        self.filter_lowpass = []
        self.filter_highpass = []
        self.filter_bandpass = []
        self.scaler = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("data-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if key == "all_negative":                   self.all_negative = value
                elif key == "sample_rate":                  self.sample_rate = value
                elif key == "filter_highpass":              self.filter_highpass = value
                elif key == "filter_lowpass":               self.filter_lowpass = value
                elif key == "filter_bandpass":              self.filter_bandpass = value
                elif key == "filter_notch":                 self.filter_notch = value
                elif key == "scaler":                       self.scaler = value
                else: sys.exit("# ERROR: invalid key [%s] in preprocess-config file" % key)

    def get_config(self):
        configs = []
        if self.all_negative is not None: configs.append(["all_negative", self.all_negative])
        configs.append(["sample_rate", self.sample_rate])
        if len(self.filter_highpass) > 0: configs.append(["filter_highpass", self.filter_highpass])
        if len(self.filter_lowpass) > 0:  configs.append(["filter_lowpass", self.filter_lowpass])
        if len(self.filter_bandpass) > 0: configs.append(["filter_bandpass", self.filter_bandpass])
        if len(self.filter_notch) > 0:    configs.append(["filter_notch", self.filter_notch])
        if self.scaler is not None:       configs.append(["scaler", self.scaler])

        return configs



# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:37:53 2022

@author: COCHE User
"""
import numpy as np
import torch
from helper_code import *
from preprocess import *
import os
import torch.utils.data as Data
from skmultilearn.model_selection import iterative_train_test_split
import random
import h5py
import pandas as pd
import wfdb
import ast
from sklearn.preprocessing import StandardScaler

def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def file_name(file_dir,file_class):  
  L=[]  
  for root, dirs, files in os.walk(file_dir): 
    for file in files: 
      if os.path.splitext(file)[1] == file_class: 
        L.append(os.path.join(root, file)) 
  return L

def load_labels_list(files="selected_labels.txt"):
    SaveList = []  
    with open(files, "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip(' ')  # 删除换行符
            SaveList.append(float(line))
    return SaveList


def load_dataset_super(dataset_name, preprocess_cfg, max_length, modelname,Norm_type,args):
    path = args.root
#    path='D:/ECG_project_root'
    os.chdir(path)
    CD_labels=load_labels_list(files="cd_labels.txt") ## SNOMED-CT codes for the labels belong to Conduction Disturbance 
    Rhythm_labels = load_labels_list(files="Rhythm_labels.txt") ## SNOMED-CT codes for the labels belong to Abnormal Rhythms 
    ST_labels = load_labels_list(files="ST_labels.txt") ## SNOMED-CT codes for the labels belong to ST/T Abnormalities
    other_labels = load_labels_list(files="other_labels.txt") ## SNOMED-CT codes for the labels belong to Other Abnormalities
    os.chdir(path+'/Cinc2021data/'+dataset_name)
    file_list_record = sorted(file_name(os.getcwd(), '.mat'))
    file_list_head = sorted(file_name(os.getcwd(), '.hea'))
    record_list = []
    label_list = []
    for i in zip(file_list_record, file_list_head):
        file_name_mat, file_name_head = i[0], i[1]
        multi_label = get_labels(load_header(file_name_head))
        # one_hot_label = np.zeros((len(selected_labels)))
        one_hot_label = np.zeros(5)
        for j in range(len(multi_label)):
            label = float(multi_label[j])
            if label in CD_labels:
                one_hot_label[0]=1
            if label in Rhythm_labels:
                one_hot_label[1]=1
            if label in ST_labels:
                one_hot_label[2]=1
            if label in other_labels:
                one_hot_label[3]=1
        if len(multi_label)==1 and float(multi_label[0])==426783006:
            one_hot_label[4]=1
        if np.sum(one_hot_label) == 0:
            continue
        if Norm_type=='channel':
            record = preprocess_signal(recording_normalize(file_name_mat, file_name_head), preprocess_cfg,
                                get_frequency(load_header(file_name_head)),max_length)
        else:
            record = recording_normalize(file_name_mat, file_name_head)
        if record.shape[1]<max_length:
            record=np.column_stack((record,np.zeros((12,max_length-record.shape[1]))))
        elif record.shape[1]>max_length:
            record = record[:,0:max_length]
        if modelname=='CNNATT':
            record=record.astype('float32')
            record_list.append(record.reshape((record.shape[0], 1, record.shape[1])))
        else:
            record_list.append(record)
        label_list.append(one_hot_label)
    return record_list, label_list


def dataset_prepare_semi(test_dataset,max_length,modelname,Norm_type,args,ifsuper=True,reprepare=True,device='cuda:0'):
    label_ratio,batch_size,unlabel_amount_coeff=args.label_ratio,args.batch_size,args.unlabel_amount_coeff
    dataset_list=['WFDB_Ga','WFDB_PTBXL','WFDB_Ningbo','WFDB_ChapmanShaoxing']
    preprocess_cfg = PreprocessConfig("preprocess.json")
    test_dataset_name=dataset_list[test_dataset]
    if reprepare: ## prepare the dataset
        record_list, label_list=load_dataset_super(test_dataset_name, preprocess_cfg,max_length, modelname,Norm_type,args)
        test_record_set=np.stack(record_list,axis=0)
        test_label_set = np.vstack(label_list)
        os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_' + test_dataset_name + '_'+modelname+'32.hdf5', 'w')
        hf.create_dataset('record_set', data=test_record_set)
        hf.create_dataset('label_set', data=test_label_set)
        hf.close()
        del record_list,label_list
    else:
        os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_'+test_dataset_name+'_'+modelname+'32.hdf5', 'r')
#        os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
#        hf = h5py.File('dataset_' + test_dataset_name + '_'+modelname+'32.hdf5', 'r')
        hf.keys()
        test_record_set = np.array(hf.get('record_set'))
        test_label_set = np.array(hf.get('label_set'))
        hf.close()
    print('test_size: after split:' + str(test_record_set.shape))
    print('test_labelsize: after split:' + str(test_label_set.shape))
    print('class_distribution')
    print(np.sum(test_label_set,axis=0))
    num_class=len(test_label_set[0])
    print(num_class)
    setup_seed(args.seed)
    train_record_set, train_label_set, test_record_set,test_label_set = iterative_train_test_split(test_record_set,test_label_set, test_size=0.2)
    valid_record_set, valid_label_set, test_record_set,test_label_set = iterative_train_test_split(test_record_set,test_label_set, test_size=0.5)
    print('valid_ratio:',np.sum(valid_label_set,axis=0)/len(valid_label_set))
    print('test_ratio:',np.sum(test_label_set,axis=0)/len(test_label_set))
    Ltrain_record_set, Ltrain_label_set, ULtrain_record_set, ULtrain_label_set = iterative_train_test_split(train_record_set,train_label_set, test_size=1-label_ratio)
    positive_weight_test=np.sum(Ltrain_label_set,axis=0)/len(Ltrain_label_set)
    print('train_size:after split:'+str(Ltrain_record_set.shape))
    
    #Ltrain_record_set,ULtrain_record_set,valid_record_set,test_record_set = ptb_style_normalization(Ltrain_record_set,ULtrain_record_set,valid_record_set,test_record_set)
    
    torch_dataset_test  = Data.TensorDataset(torch.from_numpy(test_record_set).to(device),torch.from_numpy(test_label_set).to(device),torch.from_numpy(np.arange(0,len(test_label_set))).to(device))
    torch_dataset_valid = Data.TensorDataset(torch.from_numpy(valid_record_set).to(device),torch.from_numpy(valid_label_set).to(device),torch.from_numpy(np.arange(0,len(valid_label_set))).to(device))
    torch_dataset_Ltrain = Data.TensorDataset(torch.from_numpy(Ltrain_record_set).to(device),torch.from_numpy(Ltrain_label_set).to(device),torch.from_numpy(np.arange(0,len(Ltrain_label_set))).to(device))
    torch_dataset_ULtrain = Data.TensorDataset(torch.from_numpy(ULtrain_record_set).to(device),torch.from_numpy(ULtrain_label_set).to(device),torch.from_numpy(np.arange(0,len(ULtrain_label_set))).to(device))
    loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=batch_size,shuffle=True)
    loader_Ltrain = Data.DataLoader(dataset=torch_dataset_Ltrain,batch_size=batch_size,shuffle=True)
    loader_ULtrain = Data.DataLoader(dataset=torch_dataset_ULtrain,batch_size=unlabel_amount_coeff*batch_size,shuffle=True)
    loader_valid = Data.DataLoader(dataset=torch_dataset_valid,batch_size=batch_size,shuffle=True)  
    return loader_test,loader_Ltrain,loader_ULtrain,loader_valid,num_class,positive_weight_test


def dataset_prepare_semi_crossdataset(test_dataset,max_length,modelname,Norm_type,args,ifsuper=True,device='cuda:0'):
    label_ratio,batch_size,unlabel_amount_coeff=args.label_ratio,args.batch_size,args.unlabel_amount_coeff
    dataset_list=['WFDB_Ga','WFDB_PTBXL','WFDB_Ningbo','WFDB_ChapmanShaoxing']
    preprocess_cfg = PreprocessConfig("preprocess.json")
    train_list = [0, 1, 2, 3]
    train_list.remove(test_dataset)
    sample_nums=np.array([9906,19634,34324,9717])
    train_record_set=np.zeros((sum(sample_nums[np.array(train_list)]),12,1,6144),dtype=np.float32)
    train_label_set=np.zeros((sum(sample_nums[np.array(train_list)]),args.num_of_class),dtype=np.float32)##
    flag=0
    for i in train_list:
        os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_'+dataset_list[i]+'_'+modelname+'32.hdf5', 'r')
        hf.keys()
        train_label_set[flag:flag+sample_nums[i],:]=np.array(hf.get('label_set'))
        train_record_set[flag:flag+sample_nums[i],:,:,:]=np.array(hf.get('record_set'))
        flag+=sample_nums[i]
        print('success')
        hf.close()
        del hf
    print('loading success')
    print('train_size:before split:'+str(train_record_set.shape))
    print('train_labelsize: before split:' + str(train_label_set.shape))
    index=np.arange(train_label_set.shape[0]).reshape((train_label_set.shape[0],1))
    train_idx_set, train_label_set, valid_idx_set, valid_label_set = iterative_train_test_split(index,train_label_set, test_size=0.1)
    Ltrain_idx_set, Ltrain_label_set, ULtrain_idx_set, ULtrain_label_set = iterative_train_test_split(train_idx_set,train_label_set, test_size=1-label_ratio)   
#    ## save split result
#    split_list=[Ltrain_idx_set, ULtrain_idx_set, valid_idx_set]
#    np.save('cross_split_'+dataset_list[test_dataset]+'_'+str(args.seed)+'.npy',arr=split_list)
    
    # load split result
#    split_list=np.load('cross_split_'+dataset_list[test_dataset]+'_'+str(args.seed)+'.npy',allow_pickle=True)
#    Ltrain_idx_set, ULtrain_idx_set, valid_idx_set=split_list[0],split_list[1],split_list[2]

    Ltrain_idx_set=Ltrain_idx_set.squeeze()
    ULtrain_idx_set=ULtrain_idx_set.squeeze()
    valid_idx_set=valid_idx_set.squeeze()
    Ltrain_record_set,ULtrain_record_set,valid_record_set=train_record_set[Ltrain_idx_set,:,:,:],train_record_set[ULtrain_idx_set,:,:,:],train_record_set[valid_idx_set,:,:,:]
    test_dataset_name=dataset_list[test_dataset]

    os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
    hf = h5py.File('dataset_'+test_dataset_name+'_'+modelname+'32.hdf5', 'r')
    hf.keys()
    test_record_set = np.array(hf.get('record_set'))
    test_label_set = np.array(hf.get('label_set'))
    hf.close()
    print('test_size: after split:' + str(test_record_set.shape))
    print('test_labelsize: after split:' + str(test_label_set.shape))
    print('train_size:after split:'+str(Ltrain_record_set.shape))
    num_class=len(test_label_set[0])
    print(num_class)
    positive_weight_test=np.sum(Ltrain_label_set,axis=0)/len(Ltrain_label_set)
    torch_dataset_test  = Data.TensorDataset(torch.from_numpy(test_record_set).to(device),torch.from_numpy(test_label_set).to(device),torch.from_numpy(np.arange(0,len(test_label_set))).to(device))
    torch_dataset_valid = Data.TensorDataset(torch.from_numpy(valid_record_set).to(device),torch.from_numpy(valid_label_set).to(device),torch.from_numpy(np.arange(0,len(valid_label_set))).to(device))
    torch_dataset_Ltrain = Data.TensorDataset(torch.from_numpy(Ltrain_record_set).to(device),torch.from_numpy(Ltrain_label_set).to(device),torch.from_numpy(np.arange(0,len(Ltrain_label_set))).to(device))
    torch_dataset_ULtrain = Data.TensorDataset(torch.from_numpy(ULtrain_record_set).to(device),torch.from_numpy(ULtrain_label_set).to(device),torch.from_numpy(np.arange(0,len(ULtrain_label_set))).to(device))
    loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=batch_size,shuffle=True)
    loader_Ltrain = Data.DataLoader(dataset=torch_dataset_Ltrain,batch_size=batch_size,shuffle=True)
    loader_ULtrain = Data.DataLoader(dataset=torch_dataset_ULtrain,batch_size=unlabel_amount_coeff*batch_size,shuffle=True)
    loader_valid = Data.DataLoader(dataset=torch_dataset_valid,batch_size=batch_size,shuffle=True)  
    return loader_test,loader_Ltrain,loader_ULtrain,loader_valid,num_class,positive_weight_test

def ptb_style_normalization(Ltrain_record_set,ULtrain_record_set,valid_record_set,test_record_set):
    Ltrain_record_set = np.squeeze(Ltrain_record_set, axis=2)
    ULtrain_record_set = np.squeeze(ULtrain_record_set, axis=2)
    valid_record_set = np.squeeze(valid_record_set, axis=2)
    test_record_set = np.squeeze(test_record_set, axis=2)
    
    Ltrain_record_set = np.transpose(Ltrain_record_set, (0, 2, 1))
    ULtrain_record_set = np.transpose(ULtrain_record_set, (0, 2, 1))
    valid_record_set = np.transpose(valid_record_set, (0, 2, 1))
    test_record_set = np.transpose(test_record_set, (0, 2, 1))
    
    X_scaler = StandardScaler()
    X_scaler.fit(np.vstack((Ltrain_record_set,ULtrain_record_set)).reshape(-1, Ltrain_record_set.shape[-1]))

    Ltrain_record_set = X_scaler.transform(Ltrain_record_set.reshape(-1, Ltrain_record_set.shape[-1])).reshape(Ltrain_record_set.shape)
    ULtrain_record_set = X_scaler.transform(ULtrain_record_set.reshape(-1, ULtrain_record_set.shape[-1])).reshape(ULtrain_record_set.shape)
    valid_record_set = X_scaler.transform(valid_record_set.reshape(-1, valid_record_set.shape[-1])).reshape(valid_record_set.shape)
    test_record_set = X_scaler.transform(test_record_set.reshape(-1, test_record_set.shape[-1])).reshape(test_record_set.shape)
    
    Ltrain_record_set = np.transpose(Ltrain_record_set, (0, 2, 1))
    ULtrain_record_set = np.transpose(ULtrain_record_set, (0, 2, 1))
    valid_record_set = np.transpose(valid_record_set, (0, 2, 1))
    test_record_set = np.transpose(test_record_set, (0, 2, 1))
    
    Ltrain_record_set = np.expand_dims(Ltrain_record_set, axis=2)
    ULtrain_record_set = np.expand_dims(ULtrain_record_set, axis=2)
    valid_record_set = np.expand_dims(valid_record_set, axis=2)
    test_record_set = np.expand_dims(test_record_set, axis=2)
    
    return Ltrain_record_set,ULtrain_record_set,valid_record_set,test_record_set

def dataset_prepare_semi_mixdataset(max_length,modelname,Norm_type,args,ifsuper=True,reprepare=False,device='cuda:0'):
    label_ratio,batch_size,unlabel_amount_coeff=args.label_ratio,args.batch_size,args.unlabel_amount_coeff
    dataset_list=['WFDB_Ga','WFDB_PTBXL','WFDB_Ningbo','WFDB_ChapmanShaoxing','WFDB_CPSC2018','physionet2017']
    preprocess_cfg = PreprocessConfig("preprocess.json")
    train_list=[0,1,2,3]
    os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
    sample_nums=np.array([9906,19634,34324,9717])
    train_record_set=np.zeros((sum(sample_nums[np.array(train_list)]),12,1,6144),dtype=np.float32)
    train_label_set=np.zeros((sum(sample_nums[np.array(train_list)]),args.num_of_class),dtype=np.float32)##
    print(train_list)
    flag=0
    for i in train_list:
        os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_'+dataset_list[i]+'_'+modelname+'32.hdf5', 'r')
        hf.keys()
        train_label_set[flag:flag+sample_nums[i],:]=np.array(hf.get('label_set'))
        train_record_set[flag:flag+sample_nums[i],:,:,:]=np.array(hf.get('record_set'))
        flag+=sample_nums[i]
        print('success')
        hf.close()
        del hf
    print('loading success')
    print('label distribution')
    print(np.sum(train_label_set,axis=0))
    print('train_size:before split:'+str(train_record_set.shape))
    print('train_labelsize: before split:' + str(train_label_set.shape))
    index=np.arange(train_label_set.shape[0]).reshape((train_label_set.shape[0],1))
    train_idx_set, train_label_set, valid_idx_set, valid_label_set = iterative_train_test_split(index,train_label_set, test_size=0.2)
    test_idx_set, test_label_set, valid_idx_set, valid_label_set = iterative_train_test_split(valid_idx_set, valid_label_set, test_size=0.5)
    Ltrain_idx_set, Ltrain_label_set, ULtrain_idx_set, ULtrain_label_set = iterative_train_test_split(train_idx_set,train_label_set, test_size=1-label_ratio)
    Ltrain_idx_set=Ltrain_idx_set.squeeze()
    ULtrain_idx_set=ULtrain_idx_set.squeeze()
    valid_idx_set=valid_idx_set.squeeze()
    test_idx_set=test_idx_set.squeeze()
    Ltrain_record_set,ULtrain_record_set,valid_record_set,test_record_set=train_record_set[Ltrain_idx_set,:,:,:],train_record_set[ULtrain_idx_set,:,:,:],train_record_set[valid_idx_set,:,:,:],train_record_set[test_idx_set,:,:,:]
    print('train_size:after split:'+str(Ltrain_record_set.shape))
    print('train_size:after split:'+str(Ltrain_record_set.shape))
    print('test_size: after split:' + str(test_record_set.shape))
    print('test_labelsize: after split:' + str(test_label_set.shape))
    num_class=len(test_label_set[0])
    print(num_class)
    print(test_record_set.shape)
    positive_weight_test=np.sum(Ltrain_label_set,axis=0)/len(Ltrain_label_set)
    print(np.sum(Ltrain_label_set,axis=0))
    #Ltrain_record_set,ULtrain_record_set,valid_record_set,test_record_set = ptb_style_normalization(Ltrain_record_set,ULtrain_record_set,valid_record_set,test_record_set)
    torch_dataset_test  = Data.TensorDataset(torch.from_numpy(test_record_set).to(device),torch.from_numpy(test_label_set).to(device),torch.from_numpy(np.arange(0,len(test_label_set))).to(device))
    torch_dataset_valid = Data.TensorDataset(torch.from_numpy(valid_record_set).to(device),torch.from_numpy(valid_label_set).to(device),torch.from_numpy(np.arange(0,len(valid_label_set))).to(device))
    torch_dataset_Ltrain = Data.TensorDataset(torch.from_numpy(Ltrain_record_set).to(device),torch.from_numpy(Ltrain_label_set).to(device),torch.from_numpy(np.arange(0,len(Ltrain_label_set))).to(device))
    torch_dataset_ULtrain = Data.TensorDataset(torch.from_numpy(ULtrain_record_set).to(device),torch.from_numpy(ULtrain_label_set).to(device),torch.from_numpy(np.arange(0,len(ULtrain_label_set))).to(device))
    loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=batch_size,shuffle=True)
    loader_Ltrain = Data.DataLoader(dataset=torch_dataset_Ltrain,batch_size=batch_size,shuffle=True)
    loader_ULtrain = Data.DataLoader(dataset=torch_dataset_ULtrain,batch_size=unlabel_amount_coeff*batch_size,shuffle=True)
    loader_valid = Data.DataLoader(dataset=torch_dataset_valid,batch_size=batch_size,shuffle=True)  
    return loader_test,loader_Ltrain,loader_ULtrain,loader_valid,num_class,positive_weight_test

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def PTB_labelconvert(label_str):
    one_hot=np.zeros((1,5))
    for i in range(len(label_str)):
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
    return one_hot



def dataprepare_PTB(args,max_length=6144): ## process the data from PTB-XL, use the PTB-XL superclasses
    root = args.root
    preprocess_cfg = PreprocessConfig("preprocess.json")
    path = root + '/ptb_path/ptbxl/'
    sampling_rate=500
    os.chdir(path)
    # load and convert annotation data
    index=pd.read_csv(path+'ptbxl_database_v2.csv')
    ecg_id=index['ecg_id'].tolist()
    discard_list=[]
    for i in range(max(ecg_id)):
        if (i+1) in ecg_id:
            continue
        discard_list.append(i+1)
    Y = pd.read_csv(path+'ptbxl_database_v2.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)
    print(X.shape)
#    X = X.reshape((X.shape[0],12,sampling_rate * 10))
    
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv('scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    label_list = np.zeros((X.shape[0],5))
    for i in range(X.shape[0]):
        label = Y.scp_codes[ecg_id[i]]       
        label = aggregate_diagnostic(label,agg_df)
        print(label)
        if len(label) < 1:
            label_list[i,:]=np.zeros(5)
        else:
            label = PTB_labelconvert(label)
            label_list[i,:]=label
        
    #Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)
    test_fold = 10
    valid_fold = 9
    # Train
    X_train = X[np.where(Y.strat_fold < valid_fold)]
    y_train = label_list[(Y.strat_fold < valid_fold)]
    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = label_list[Y.strat_fold == test_fold]  
    #Valid 
    X_valid = X[np.where(Y.strat_fold == valid_fold)]
    y_valid = label_list[Y.strat_fold == valid_fold]
    
    X_scaler = StandardScaler()
    X_scaler.fit(X_train.reshape(-1, X_train.shape[-1]))

    X_train = X_scaler.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_valid = X_scaler.transform(X_valid.reshape(-1, X_valid.shape[-1])).reshape(X_valid.shape)
    X_test  = X_scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    print(X_test.shape)
    print(y_test.shape)
    
    print(X_train.shape)
    print(y_train.shape)
    
    print(X_valid.shape)
    print(y_valid.shape)
    
    record_list_train = np.zeros((X_train.shape[0],12,1,max_length)).astype('float32')
    record_list_valid = np.zeros((X_valid.shape[0],12,1,max_length)).astype('float32')
    record_list_test = np.zeros((X_test.shape[0],12,1,max_length)).astype('float32')
    #print(X[0, 0, :])
    for i in range(X_train.shape[0]):
        #print(X[i,:,:])
        #record = preprocess_signal(X[i,:,:], preprocess_cfg, sampling_rate,max_length=max_length)
        record = X_train[i,:,:].T
        record = np.column_stack((record,np.zeros((12,max_length-record.shape[1]))))
        record = record.reshape((1,12,1,max_length))
        record_list_train[i,:,:,:] = record.astype('float32')

    for i in range(X_valid.shape[0]):       
        record = X_valid[i,:,:].T
        record = np.column_stack((record,np.zeros((12,max_length-record.shape[1]))))
        record = record.reshape((1,12,1,max_length))
        record_list_valid[i,:,:,:] = record.astype('float32')
      
    for i in range(X_test.shape[0]): 
        record = X_test[i,:,:].T
        record = np.column_stack((record,np.zeros((12,max_length-record.shape[1]))))
        record = record.reshape((1,12,1,max_length))
        record_list_test[i,:,:,:] = record.astype('float32')

    os.chdir(root+'/Cinc2021data/Preprocessed_dataset')
    hf = h5py.File('trainset_' + 'PTB_ori' + '_'+'32.hdf5', 'w')
    hf.create_dataset('record_set', data=record_list_train)
    hf.create_dataset('label_set', data=y_train)
    hf.close()
    
    os.chdir(root+'/Cinc2021data/Preprocessed_dataset')
    hf = h5py.File('testset_' + 'PTB_ori' + '_'+'32.hdf5', 'w')
    hf.create_dataset('record_set', data=record_list_test)
    hf.create_dataset('label_set', data=y_test)
    hf.close()
    
    os.chdir(root+'/Cinc2021data/Preprocessed_dataset')
    hf = h5py.File('validset_' + 'PTB_ori' + '_'+'32.hdf5', 'w')
    hf.create_dataset('record_set', data=record_list_valid)
    hf.create_dataset('label_set', data=y_valid)
    hf.close()
    
def aggregate_diagnostic(y_dic,agg_df):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))


def dataloading_PTB(args):
    label_ratio,batch_size,unlabel_amount_coeff,device=args.label_ratio,args.batch_size,args.unlabel_amount_coeff,args.device
    os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
    hf = h5py.File('trainset_' + 'PTB_ori' + '_'+'32.hdf5', 'r')
    hf.keys()
    train_record_set = np.array(hf.get('record_set'))
    train_label_set = np.array(hf.get('label_set'))
    hf.close()
    print(train_record_set.shape)
    os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
    hf = h5py.File('testset_' + 'PTB_ori' + '_'+'32.hdf5', 'r')
    hf.keys()
    test_record_set = np.array(hf.get('record_set'))
    test_label_set = np.array(hf.get('label_set'))
    hf.close()
    print(test_record_set.shape)
    os.chdir(args.root+'/Cinc2021data/Preprocessed_dataset')
    hf = h5py.File('validset_' + 'PTB_ori' + '_'+'32.hdf5', 'r')
    hf.keys()
    valid_record_set = np.array(hf.get('record_set'))
    valid_label_set = np.array(hf.get('label_set'))
    hf.close()
    print(valid_record_set.shape)
    
    index=np.arange(train_label_set.shape[0]).reshape((train_label_set.shape[0],1))
    train_idx_set = index
    Ltrain_idx_set, Ltrain_label_set, ULtrain_idx_set, ULtrain_label_set = iterative_train_test_split(train_idx_set,train_label_set, test_size=1-label_ratio)
    
    Ltrain_idx_set=Ltrain_idx_set.squeeze()
    ULtrain_idx_set=ULtrain_idx_set.squeeze()
    num_class=len(test_label_set[0])
    Ltrain_record_set,ULtrain_record_set=train_record_set[Ltrain_idx_set,:,:,:],train_record_set[ULtrain_idx_set,:,:,:]
    positive_weight_test=np.sum(Ltrain_label_set,axis=0)/len(Ltrain_label_set)
    torch_dataset_test  = Data.TensorDataset(torch.from_numpy(test_record_set).to(device),torch.from_numpy(test_label_set).to(device),torch.from_numpy(np.arange(0,len(test_label_set))).to(device))
    torch_dataset_valid = Data.TensorDataset(torch.from_numpy(valid_record_set).to(device),torch.from_numpy(valid_label_set).to(device),torch.from_numpy(np.arange(0,len(valid_label_set))).to(device))
    torch_dataset_Ltrain = Data.TensorDataset(torch.from_numpy(Ltrain_record_set).to(device),torch.from_numpy(Ltrain_label_set).to(device),torch.from_numpy(np.arange(0,len(Ltrain_label_set))).to(device))
    torch_dataset_ULtrain = Data.TensorDataset(torch.from_numpy(ULtrain_record_set).to(device),torch.from_numpy(ULtrain_label_set).to(device),torch.from_numpy(np.arange(0,len(ULtrain_label_set))).to(device))
    loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=batch_size,shuffle=True)
    loader_Ltrain = Data.DataLoader(dataset=torch_dataset_Ltrain,batch_size=batch_size,shuffle=True)
    loader_ULtrain = Data.DataLoader(dataset=torch_dataset_ULtrain,batch_size=unlabel_amount_coeff*batch_size,shuffle=True)
    loader_valid = Data.DataLoader(dataset=torch_dataset_valid,batch_size=batch_size,shuffle=True)
    return loader_test,loader_Ltrain,loader_ULtrain,loader_valid,num_class,positive_weight_test


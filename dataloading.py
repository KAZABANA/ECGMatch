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
    SaveList = []  # 存档列表
    # 读取文本内容到列表
    with open(files, "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip(' ')  # 删除换行符
            SaveList.append(float(line))
    return SaveList


def load_dataset_super(dataset_name, preprocess_cfg, max_length, modelname,Norm_type):
    path='/home/coche/RushuangZhou_Phdstudent'
#    path='D:/ECG_project_root'
    os.chdir(path)
    CD_labels=load_labels_list(files="cd_labels.txt")
    Rhythm_labels = load_labels_list(files="Rhythm_labels.txt")
    ST_labels = load_labels_list(files="ST_labels.txt")
    other_labels = load_labels_list(files="other_labels.txt")
    os.chdir(path+'/Cinc2021data/'+dataset_name)
#    os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/'+dataset_name)
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
    if reprepare:
        record_list, label_list=load_dataset_super(test_dataset_name, preprocess_cfg,max_length, modelname,Norm_type)
        test_record_set=np.stack(record_list,axis=0)
        test_label_set = np.vstack(label_list)
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_' + test_dataset_name + '_'+modelname+'32.hdf5', 'w')
        hf.create_dataset('record_set', data=test_record_set)
        hf.create_dataset('label_set', data=test_label_set)
        hf.close()
        del record_list,label_list
    else:
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_' + test_dataset_name + '_'+modelname+'32.hdf5', 'r')
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
    torch_dataset_test  = Data.TensorDataset(torch.from_numpy(test_record_set).to(device),torch.from_numpy(test_label_set).to(device),torch.from_numpy(np.arange(0,len(test_label_set))).to(device))
    torch_dataset_valid = Data.TensorDataset(torch.from_numpy(valid_record_set).to(device),torch.from_numpy(valid_label_set).to(device),torch.from_numpy(np.arange(0,len(valid_label_set))).to(device))
    torch_dataset_Ltrain = Data.TensorDataset(torch.from_numpy(Ltrain_record_set).to(device),torch.from_numpy(Ltrain_label_set).to(device),torch.from_numpy(np.arange(0,len(Ltrain_label_set))).to(device))
    torch_dataset_ULtrain = Data.TensorDataset(torch.from_numpy(ULtrain_record_set).to(device),torch.from_numpy(ULtrain_label_set).to(device),torch.from_numpy(np.arange(0,len(ULtrain_label_set))).to(device))
    loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=batch_size,shuffle=True)
    loader_Ltrain = Data.DataLoader(dataset=torch_dataset_Ltrain,batch_size=batch_size,shuffle=True)
    loader_ULtrain = Data.DataLoader(dataset=torch_dataset_ULtrain,batch_size=unlabel_amount_coeff*batch_size,shuffle=True)
    loader_valid = Data.DataLoader(dataset=torch_dataset_valid,batch_size=batch_size,shuffle=True)  
    return loader_test,loader_Ltrain,loader_ULtrain,loader_valid,num_class,positive_weight_test


def dataset_prepare_semi_crossdataset(test_dataset,max_length,modelname,Norm_type,args,ifsuper=True,reprepare=False,device='cuda:0'):
    label_ratio,batch_size,unlabel_amount_coeff=args.label_ratio,args.batch_size,args.unlabel_amount_coeff
    dataset_list=['WFDB_Ga','WFDB_PTBXL','WFDB_Ningbo','WFDB_ChapmanShaoxing']
    preprocess_cfg = PreprocessConfig("preprocess.json")
    traindata_list,trainlabel_list=[],[]
    train_list = [0, 1, 2, 3]
    train_list.remove(test_dataset)
    if reprepare:
        for i in train_list:
            dataset_name=dataset_list[i]
            print(dataset_name)
            record_list, label_list=load_dataset_super(dataset_name, preprocess_cfg,max_length, modelname,Norm_type)
            traindata_list=traindata_list+record_list
            trainlabel_list=trainlabel_list+label_list
        train_record_set=np.stack(traindata_list)
        train_label_set=np.vstack(trainlabel_list)
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_all2'+dataset_list[test_dataset]+'_'+modelname+'.hdf5', 'w')
        hf.create_dataset('record_set', data=train_record_set)
        hf.create_dataset('label_set', data=train_label_set)
        hf.close()
        del traindata_list, trainlabel_list, record_list, label_list,hf
    else:
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/Preprocessed_dataset')
        sample_nums=np.array([9906,19634,34324,9717])
        train_record_set=np.zeros((sum(sample_nums[np.array(train_list)]),12,1,6144),dtype=np.float32)
        train_label_set=np.zeros((sum(sample_nums[np.array(train_list)]),5),dtype=np.float32)##
        flag=0
        for i in train_list:
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
    if reprepare:
        record_list, label_list=load_dataset_super(test_dataset_name, preprocess_cfg,max_length, modelname,Norm_type)
        test_record_set=np.stack(record_list,axis=0)
        test_label_set = np.vstack(label_list)
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_' + test_dataset_name + '_'+modelname+'32.hdf5', 'w')
        hf.create_dataset('record_set', data=test_record_set)
        hf.create_dataset('label_set', data=test_label_set)
        hf.close()
        del record_list,label_list
    else:
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_' + test_dataset_name + '_'+modelname+'32.hdf5', 'r')
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

def dataset_prepare_semi_mixdataset(max_length,modelname,Norm_type,args,ifsuper=True,reprepare=False,device='cuda:0'):
    label_ratio,batch_size,unlabel_amount_coeff=args.label_ratio,args.batch_size,args.unlabel_amount_coeff
    dataset_list=['WFDB_Ga','WFDB_PTBXL','WFDB_Ningbo','WFDB_ChapmanShaoxing','WFDB_CPSC2018','physionet2017']
    preprocess_cfg = PreprocessConfig("preprocess.json")
    traindata_list,trainlabel_list=[],[]
    print(reprepare)
    if reprepare:
        for i in range(4):
            dataset_name=dataset_list[i]
            record_list, label_list=load_dataset_super(dataset_name, preprocess_cfg,max_length, modelname,Norm_type)
            traindata_list=traindata_list+record_list
            trainlabel_list=trainlabel_list+label_list
        train_record_set=np.stack(traindata_list)
        train_label_set=np.vstack(trainlabel_list)
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/Preprocessed_dataset')
        hf = h5py.File('dataset_all2'+'_'+modelname+'.hdf5', 'w')
        hf.create_dataset('record_set', data=train_record_set)
        hf.create_dataset('label_set', data=train_label_set)
        hf.close()
        del traindata_list, trainlabel_list, record_list, label_list,hf
    else:
        train_list=[0,1,2,3]
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/Preprocessed_dataset')
        sample_nums=np.array([9906,19634,34324,9717])
        train_record_set=np.zeros((sum(sample_nums[np.array(train_list)]),12,1,6144),dtype=np.float32)
        train_label_set=np.zeros((sum(sample_nums[np.array(train_list)]),5),dtype=np.float32)##
        print(train_list)
        flag=0
        for i in train_list:
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
    torch_dataset_test  = Data.TensorDataset(torch.from_numpy(test_record_set).to(device),torch.from_numpy(test_label_set).to(device),torch.from_numpy(np.arange(0,len(test_label_set))).to(device))
    torch_dataset_valid = Data.TensorDataset(torch.from_numpy(valid_record_set).to(device),torch.from_numpy(valid_label_set).to(device),torch.from_numpy(np.arange(0,len(valid_label_set))).to(device))
    torch_dataset_Ltrain = Data.TensorDataset(torch.from_numpy(Ltrain_record_set).to(device),torch.from_numpy(Ltrain_label_set).to(device),torch.from_numpy(np.arange(0,len(Ltrain_label_set))).to(device))
    torch_dataset_ULtrain = Data.TensorDataset(torch.from_numpy(ULtrain_record_set).to(device),torch.from_numpy(ULtrain_label_set).to(device),torch.from_numpy(np.arange(0,len(ULtrain_label_set))).to(device))
    loader_test = Data.DataLoader(dataset=torch_dataset_test,batch_size=batch_size,shuffle=True)
    loader_Ltrain = Data.DataLoader(dataset=torch_dataset_Ltrain,batch_size=batch_size,shuffle=True)
    loader_ULtrain = Data.DataLoader(dataset=torch_dataset_ULtrain,batch_size=unlabel_amount_coeff*batch_size,shuffle=True)
    loader_valid = Data.DataLoader(dataset=torch_dataset_valid,batch_size=batch_size,shuffle=True)  
    return loader_test,loader_Ltrain,loader_ULtrain,loader_valid,num_class,positive_weight_test
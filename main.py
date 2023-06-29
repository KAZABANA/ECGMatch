# This is a sample Python script.
import numpy as np
import torch
from helper_code import *
from preprocess import *
import os
import random
from training_code import *
from dataloading import dataset_prepare_semi,dataset_prepare_semi_crossdataset,dataset_prepare_semi_mixdataset
def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def semi_pipeline_main(args):
    os.chdir('/home/coche/RushuangZhou_Phdstudent')
    modelname = 'CNNATT'
    seed = args.seed
    Norm_type = 'channel'
    setup_seed(seed)
    local_path = os.getcwd()
    max_length = 6144  # 4096,6144,8192
    datasetlist = [0, 1, 2, 3]
    datasetname = ['WFDB_Ga', 'WFDB_PTBXL', 'WFDB_Ningbo', 'WFDB_ChapmanShaoxing']
    semi_config = args.semi_config
    test_list=[]
    for i in range(4):
        reprepare = False
        os.chdir(local_path)
        current_name = datasetname[i]
        args.dataset=current_name
        datasetlist.remove(i)
        print(i)
        print(datasetlist)
        device = args.device
        if args.experiment=='within_dataset':
            loader_test, loader_Ltrain, loader_ULtrain, loader_valid, num_class, _ = dataset_prepare_semi(
            i, max_length, modelname, Norm_type,args,ifsuper=True, reprepare=reprepare, device=device)
        elif args.experiment=='cross_dataset':
            loader_test, loader_Ltrain, loader_ULtrain, loader_valid, num_class, _ = dataset_prepare_semi_crossdataset(
            i, max_length, modelname, Norm_type, args,ifsuper=True, reprepare=reprepare, device=device)         
        model= model_prepare(num_class)
        model.to(device)
        _, retrain_test_result, retrain_L_list, retrain_UL_list, retrain_valid_list = semi_supervised_learning_ECGmatch(
                current_name, semi_config, model, loader_Ltrain, loader_ULtrain, loader_valid, loader_test,
                device, args)
        del loader_test, loader_Ltrain, loader_ULtrain, loader_valid, num_class
        torch.cuda.empty_cache()
        print('running success')
        datasetlist = [0, 1, 2, 3]
        test_list.append(retrain_test_result)
        print('save file')
        print(retrain_test_result)
    return test_list

def semi_pipeline_main_mix(args):
    os.chdir('/home/coche/RushuangZhou_Phdstudent')
    modelname = 'CNNATT'
    seed = args.seed
    Norm_type = 'channel'
    setup_seed(seed)
    local_path = os.getcwd()
    max_length = 6144  # 4096,6144,8192
    semi_config = args.semi_config
    device=args.device
    reprepare = False
    os.chdir(local_path)
    total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    loader_test, loader_Ltrain, loader_ULtrain, loader_valid, num_class, _ = dataset_prepare_semi_mixdataset(max_length,modelname,Norm_type,args,ifsuper=True,
                                                                                                                                reprepare=reprepare,device=device)
    model= model_prepare(num_class)
    model.to(device)
    current_name='All'
    args.dataset=current_name
    _, retrain_test_result, retrain_L_list, retrain_UL_list, retrain_valid_list = semi_supervised_learning_ECGmatch(
                current_name, semi_config, model, loader_Ltrain, loader_ULtrain, loader_valid, loader_test,
                device, args)
    del loader_test, loader_Ltrain, loader_ULtrain, loader_valid, num_class
    print('save file')
    print(retrain_test_result)
    return retrain_test_result


def exp_withindataset(ratio,seed):
    os.chdir('/home/coche/RushuangZhou_Phdstudent')
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--metrics', type=str, default='Map_value')
    parser.add_argument('--similarity', type=str, default='cosine')
    parser.add_argument('--experiment', type=str, default='within_dataset')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--teacher_pretrain', type=bool, default=True)
    parser.add_argument('--Pretrain_epoch', type=int, default=int(200*(0.1/ratio)))
    parser.add_argument('--num_of_class', type=float, default=5)
    parser.add_argument('--Semitrain_epoch', type=int, default=int(200*(0.1/ratio)))
    parser.add_argument('--Semi_optimizer', type=str, default='SGD')
    parser.add_argument('--Semi_lr_decay', type=float, default=5e-4)
    parser.add_argument('--Semi_lr_rate', type=float, default=3e-2)#3e-2
    parser.add_argument('--Semi_momentum', type=float, default=0.9)
    parser.add_argument('--ema', type=bool, default=False)  
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--Semi_loss_weight', type=float, default=0.1)
    parser.add_argument('--batch_size', type=float, default=64)
    parser.add_argument('--unlabel_amount_coeff', type=float, default=7)
    parser.add_argument('--complexity', type=float, default=128)
    parser.add_argument('--label_ratio', type=float, default=ratio)
    parser.add_argument('--seed', type=int, default=seed)# default 20
    parser.add_argument('--model_config', type=str, default='baseline_noweightnorm_nonecknorm')
    parser.add_argument('--semi_config', type=str, default='_ECGmatch')
    parser.add_argument('--label_relationship', type=str, default=True)
    parser.add_argument('--neighbor_weight', type=str, default=True)
    parser.add_argument('--relationship_weight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:3')
    args = parser.parse_args()
    print(args.label_ratio)
    print(args.Pretrain_epoch)
    print(args.Semitrain_epoch)
    print(args.Semi_optimizer)
    setup_seed(args.seed)
    ratio=str(args.label_ratio)
    ratio=ratio.replace('.', '')
    unlabel=str(args.unlabel_amount_coeff)
    batch=str(args.batch_size)
    seed=args.seed
    print('seed:',seed)
#    # ECGMatch
    # sensitivity analysis for two hyper-parameters: lambda_{u} and lambda_{f}
    test_list=[]
    args.ema=False
    if args.ema==True:
        ema_flag='ema'
    else:
        ema_flag='noema'
    print(ema_flag)
    args.semi_config = '_ECGmatch'
    for i in range(5):
        for j in range(5):
            print(args.similarity)
            args.semi_config = '_ECGmatch'     
            args.Semi_loss_weight = 2*i/5 # lambda_{u}
            args.relationship_weight = 2*j/5 # lambda_{f}
            args.batch_size=64
            args.unlabel_amount_coeff=7
            print(args.Semi_loss_weight)
            print(args.relationship_weight)
            os.chdir('/home/coche/RushuangZhou_Phdstudent')
            test_list.append(semi_pipeline_main(args))
    os.chdir('/home/coche/RushuangZhou_Phdstudent/result')
    np.save(args.model_config + args.semi_config + '_grid_search_within_dataset'+'_ratio'+ratio+'_default_'+ema_flag+'_batch'+str(batch)+'_'+unlabel+'unlabel'+'_seed'+str(seed)+'.npy', test_list)
    
def exp_crossdataset(ratio,seed):
    os.chdir('/home/coche/RushuangZhou_Phdstudent')
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--teacher_pretrain', type=bool, default=True)
    parser.add_argument('--metrics', type=str, default='Map_value')
    parser.add_argument('--similarity', type=str, default='cosine')
    parser.add_argument('--experiment', type=str, default='cross_dataset')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--Pretrain_epoch', type=int, default=int(200*(0.1/ratio)))
    parser.add_argument('--num_of_class', type=float, default=5)
    parser.add_argument('--Semitrain_epoch', type=int, default=int(200*(0.1/ratio)))
    parser.add_argument('--Semi_optimizer', type=str, default='SGD')
    parser.add_argument('--Semi_lr_decay', type=float, default=5e-4)
    parser.add_argument('--Semi_lr_rate', type=float, default=3e-2)
    parser.add_argument('--ema', type=bool, default=False)
    parser.add_argument('--Semi_momentum', type=float, default=0.9)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--Semi_loss_weight', type=float, default=0.1)
    parser.add_argument('--batch_size', type=float, default=64)
    parser.add_argument('--unlabel_amount_coeff', type=float, default=2)
    parser.add_argument('--complexity', type=float, default=128)
    parser.add_argument('--label_ratio', type=float, default=ratio)
    parser.add_argument('--seed', type=int, default=seed)#default 20
    parser.add_argument('--model_config', type=str, default='baseline_noweightnorm_nonecknorm')
    parser.add_argument('--semi_config', type=str, default='_ECGmatch')
    parser.add_argument('--label_relationship', type=str, default=True)
    parser.add_argument('--neighbor_weight', type=str, default=True)
    parser.add_argument('--relationship_weight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:3')

    args = parser.parse_args()
    print(args.label_ratio)
    print(args.Pretrain_epoch)
    print(args.Semitrain_epoch)
    print(args.Semi_optimizer)
    setup_seed(args.seed)
    ratio=str(args.label_ratio)
    ratio=ratio.replace('.', '')
    batch=str(args.batch_size)
    seed=args.seed
    print('seed:',seed)
    
    args.unlabel_amount_coeff=2
    unlabel=str(args.unlabel_amount_coeff)
     #ECGMatch
    args.ema=False
    if args.ema==True:
        ema_flag='ema'
    else:
        ema_flag='noema'
    print(ema_flag)
    test_list = []
    args.semi_config = '_ECGmatch'
    for i in range(5):
        for j in range(5):
            args.semi_config = '_ECGmatch'
            args.Semi_loss_weight = 2*i/5
            args.relationship_weight = 2*j/5
            args.batch_size=64
            args.unlabel_amount_coeff=2
            print(args.Semi_loss_weight)
            print(args.relationship_weight)
            os.chdir('/home/coche/RushuangZhou_Phdstudent')
            test_list.append(semi_pipeline_main(args))
    os.chdir('/home/coche/RushuangZhou_Phdstudent/result')
    np.save(args.model_config + args.semi_config + '_grid_search_cross_dataset'+'_ratio'+ratio+'_default_'+ema_flag+'_batch'+str(batch)+'_'+unlabel+'unlabel'+'_seed'+str(seed)+'.npy', test_list)

def exp_mixdataset(ratio,seed):
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--teacher_pretrain', type=bool, default=True)
    parser.add_argument('--metrics', type=str, default='Map_value')
    parser.add_argument('--similarity', type=str, default='cosine')
    parser.add_argument('--experiment', type=str, default='mix_dataset')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--Pretrain_epoch', type=int, default=int(200*(0.1/ratio)))
    parser.add_argument('--num_of_class', type=float, default=5)
    parser.add_argument('--Semitrain_epoch', type=int, default=int(200*(0.1/ratio)))
    parser.add_argument('--Semi_optimizer', type=str, default='SGD')
    parser.add_argument('--Semi_lr_decay', type=float, default=5e-4)
    parser.add_argument('--Semi_lr_rate', type=float, default=3e-2)
    parser.add_argument('--ema', type=bool, default=False)
    parser.add_argument('--Semi_momentum', type=float, default=0.9)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--Semi_loss_weight', type=float, default=0.1)
    parser.add_argument('--batch_size', type=float, default=64)
    parser.add_argument('--unlabel_amount_coeff', type=float, default=2)
    parser.add_argument('--complexity', type=float, default=128)
    parser.add_argument('--label_ratio', type=float, default=ratio)
    parser.add_argument('--seed', type=int, default=seed)#default 20
    parser.add_argument('--model_config', type=str, default='baseline_noweightnorm_nonecknorm')
    parser.add_argument('--semi_config', type=str, default='_ECGmatch')
    parser.add_argument('--label_relationship', type=str, default=True)
    parser.add_argument('--neighbor_weight', type=str, default=True)
    parser.add_argument('--relationship_weight', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda:3')

    args = parser.parse_args()
    print(args.label_ratio)
    print(args.Pretrain_epoch)
    print(args.Semitrain_epoch)
    print(args.Semi_optimizer)
    setup_seed(args.seed)
    ratio=str(args.label_ratio)
    ratio=ratio.replace('.', '')
    unlabel=str(args.unlabel_amount_coeff)
    batch=str(args.batch_size)
    seed=args.seed
    print('seed:',seed)

     #ECGMatch
    args.ema=False
    if args.ema==True:
        ema_flag='ema'
    else:
        ema_flag='noema'
    print(ema_flag)
    test_list = []
    args.strong_neighbor = True
    args.weak_neighbor = False
    args.teacher = True
    args.semi_config = '_ECGmatch'
    for i in range(5):
        for j in range(5):
            args.semi_config = '_ECGmatch'
            args.Semi_loss_weight = 2*i/5
            args.relationship_weight = 2*j/5
            args.batch_size=64
            args.unlabel_amount_coeff=2
            print(args.Semi_loss_weight)
            print(args.relationship_weight)
            os.chdir('/home/coche/RushuangZhou_Phdstudent')
            test_list.append(semi_pipeline_main_mix(args))
    os.chdir('/home/coche/RushuangZhou_Phdstudent/result')
    np.save(args.model_config + args.semi_config + '_grid_search_mix_dataset'+'_ratio'+ratio+'_default_'+ema_flag+'_batch'+str(batch)+'_'+unlabel+'unlabel'+'_seed'+str(seed)+'.npy', test_list)

if __name__ == '__main__':
    seed=19
    exp_withindataset(0.05,seed)
    exp_crossdataset(0.01,seed)
    exp_mixdataset(0.01,seed)
    
    seed=20
    exp_withindataset(0.05,seed)
    exp_crossdataset(0.01,seed)
    exp_mixdataset(0.01,seed)
    
    seed=21
    exp_withindataset(0.05,seed)
    exp_crossdataset(0.01,seed)
    exp_mixdataset(0.01,seed)
    
    
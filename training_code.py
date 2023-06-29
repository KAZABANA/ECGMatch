# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 11:58:27 2022

@author: COCHE User
"""
import numpy as np
import torch
import torch.nn.functional as F
from helper_code import *
from preprocess import *
import os
from model_code import NN,ModelEma,tar_augmentation,StepwiseLR
from tqdm import tqdm
from pytorchtools import EarlyStopping
from evaluation import print_result,find_thresholds
from AsymmetricLoss import AsymmetricLoss_dynamic

def model_prepare(num_class=5):
    num_leads=12
    model=NN(nOUT=num_class,complexity=128,inputchannel=num_leads)
    # optimizer=torch.optim.Adam(model.parameters(), lr=0.01)
    return model

def validate(model, valloader,device,iftest=False,threshold=0.5*np.ones(5),iftrain=False,args=None):
    model.eval()
    losses, probs, lbls,logit = [], [], [],[]
    for i, (inp_windows_t, lbl_t,_) in tqdm(enumerate(valloader), total=len(valloader)):
        inp_windows_t, lbl_t = inp_windows_t.float(), lbl_t.int()
        with torch.no_grad():
            out, _ = model(inp_windows_t)
            loss = F.binary_cross_entropy_with_logits(out, lbl_t.float())
            prob = out.sigmoid().data.cpu().numpy()
            losses.append(loss.item())
            probs.append(prob)
            lbls.append(lbl_t.data.cpu().numpy())
            logit.append(out.data.cpu().numpy())
    loss = np.mean(losses)
    lbls = np.concatenate(lbls)
    probs = np.concatenate(probs)
    logit = np.concatenate(logit)
    if iftest:
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'test', threshold)
    elif iftrain:
        threshold=find_thresholds(lbls.copy(), probs.copy())
        valid_result = print_result(np.mean(losses), lbls.copy(), probs.copy(), 'train',threshold)
    else:
        threshold=find_thresholds(lbls, probs)
        valid_result = print_result(np.mean(losses), lbls, probs, 'valid',threshold)
    neg_ratio=(len(probs)-np.sum(probs,axis=0))/np.sum(probs,axis=0)
    print('pred_ratio',neg_ratio)
    valid_result.update({'neg_ratio':neg_ratio})
    valid_result.update({'threshold':threshold})
    return valid_result

def similarity_metrics(score, metrics):
    eps=0# eps=0 in previous experiment
    if metrics == 'cosine':
        # calculate cosine similarity
        sim_mat = torch.matmul(score.T, score) / (torch.matmul(torch.norm(score, dim=0).reshape(-1, 1), torch.norm(score, dim=0).reshape(1, -1))+eps)
    elif metrics == 'euclid':
        # calculate Euclidean distance
        sim_mat = torch.cdist(score.T, score.T)
        sim_mat=1/(1+sim_mat)
    elif metrics == 'pearson':
        # calculate Pearson correlation coefficient
        score_centered = score - score.mean(dim=0, keepdim=True)
        sim_mat = (torch.matmul(score_centered.T, score_centered) / torch.matmul(torch.norm(score_centered, dim=0).reshape(-1, 1), torch.norm(score_centered, dim=0).reshape(1, -1)))**2
    else:
        raise ValueError("Invalid similarity metric specified.")
    return sim_mat
   
def semi_supervised_learning_ECGmatch(dataset_name,semi_config,model,loader_train,loader_ULtrain,loader_valid,loader_test,device,args):
    if args.experiment=='cross_dataset':
        dataset_name='cross_'+dataset_name
    if args.experiment=='mix_dataset' :
        dataset_name='mix_'+dataset_name
    if args.experiment=='within_dataset':
        dataset_name='within_'+dataset_name
        
    ## load the pretrained teacher model (you can pretrain the teacher by setting teacher_pretrain==False and Semi_loss_weight=0,relationship_weight=0)
    if args.teacher_pretrain:
        print('load_teacher')
        os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/result/pretrain_model')
        model.load_state_dict(torch.load(dataset_name +semi_config+'_'+str(int(0))+'_'+str(int(0))+'_'+'noema'+'_checkpoint_'+str(args.seed)+'_pre.pkl',map_location={'cuda:1': args.device}))
    loss_save='/home/coche/RushuangZhou_Phdstudent/Cinc2021data/result/run_log/log_p.txt'
    file_save=open(loss_save,mode='a')
    file_save.write('\n'+'Semi_loss_weight:'+str(args.Semi_loss_weight)+'_relation_loss_weight:'+str(args.relationship_weight))    
    file_save.close()
    print('save_log')
    if args.ema:
        ema_flag='ema'
    else:
        ema_flag='noema'   
    batch_size=args.batch_size
    unlabel_amount_coeff=args.unlabel_amount_coeff
    print(args.Semi_loss_weight)
    print(args.relationship_weight)
    if args.Semi_optimizer=='Adam':
        optimizer=torch.optim.Adam(model.get_parameters(),lr=args.Semi_lr_rate,weight_decay=args.Semi_lr_decay)
    elif args.Semi_optimizer=='SGD':
        optimizer=torch.optim.SGD(model.get_parameters(),lr=args.Semi_lr_rate,weight_decay=args.Semi_lr_decay,momentum=args.Semi_momentum)
    model.to(device)
    iteration = len(loader_train) * args.Semitrain_epoch
    lr_scheduler = StepwiseLR(optimizer, init_lr=args.Semi_lr_rate, gamma=10, decay_rate=0.75, max_iter=iteration)
    criterion = AsymmetricLoss_dynamic(gamma_neg=0, gamma_pos=0, alpha=np.zeros(1),clip=0.00, disable_torch_grad_focal_loss=True)
    ## If the class distribution is very imbalance, you can tune the gamma; but in our experiments, we set all the gamma as zero
    ## in this case, the AsymmetricLoss_dynamic loss is the same as binary cross entropy 
    train_result_list, ULtrain_result_list,valid_result_list = [], [],[]
    label_iter = iter(loader_train)
    unlabel_iter = iter(loader_ULtrain)
    K=args.K
    dataset_name=dataset_name+'testing'
    dataset_name=dataset_name+args.similarity
    early_stopping = EarlyStopping(20, verbose=True,dataset_name=dataset_name,model_cofig=semi_config,delta=0,args=args)#15
    
    ## feature and prediction banks initialization
    fea_bank_L,fea_bank_UL = torch.randn(len(loader_train.dataset), 128),torch.randn(len(loader_ULtrain.dataset), 128)
    score_bank_L,score_bank_UL = torch.randn(len(loader_train.dataset), args.num_of_class).to(device),torch.randn(len(loader_ULtrain.dataset), args.num_of_class).to(device)
    EMA_model=ModelEma(model)
    with torch.no_grad():
        iter_labeled = iter(loader_train)
        for i in range(len(loader_train)):
            model.eval()
            data = iter_labeled .next()
            label  = data[1].float()
            indx = data[-1].detach().clone().cpu().numpy()
            score_bank_L[indx] = label.detach().clone()  # .cpu()
        relationship_mat_L=similarity_metrics(score_bank_L, args.similarity)
        print(relationship_mat_L)
        relationship_mat_L=torch.clip(relationship_mat_L,0,1)
    with torch.no_grad():
        iter_unlabeled = iter(loader_ULtrain)
        for i in range(len(loader_ULtrain)):
            model.eval()
            data = iter_unlabeled.next()
            inputs = data[0].float()
            indx = data[-1].detach().clone().cpu().numpy()
            output = model.bottleneck(model.feature_extraction(inputs))
            output_norm = F.normalize(output)
            outputs = model.classifier(output)
            outputs = outputs.sigmoid()
            fea_bank_UL[indx] = output_norm.detach().clone().cpu()
            score_bank_UL[indx] = outputs.detach().clone()  # .cpu()
    unlabel_size=batch_size*unlabel_amount_coeff
    print('\n')
    print('semi supervised training') 
    mem_used = torch.cuda.max_memory_allocated(device=device) / 1024 / 1024  # 转换为 MB
    print(f"GPU {device}: {mem_used:.2f} MB")
    
    ## training pipeline
    for current in range(iteration): 
        model_teacher=EMA_model.module
        
        ## model evaluation and early stopping
        if current % 50 == 0:
            print('current iter',current)
            print('training')
            if args.ema:
                training_result=validate(model_teacher, loader_train, device,iftrain=True)
                valid_result = validate(model_teacher, loader_valid, device)
                if current>0:
                    early_stopping(1/valid_result[args.metrics], model_teacher)
                    train_result_list.append(training_result)
                    valid_result_list.append(valid_result)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
            else:
                training_result=validate(model, loader_train, device,iftrain=True)
                valid_result = validate(model, loader_valid, device)
                if current>0:
                    early_stopping(1/valid_result[args.metrics], model)
                    train_result_list.append(training_result)
                    valid_result_list.append(valid_result)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
        ## mini-batch sampling
        try:
            source_data, source_label,src_idx = next(label_iter)
        except Exception as err:
            label_iter = iter(loader_train)
            source_data, source_label,src_idx = next(label_iter)
        try:
            target_data, _, tar_idx = next(unlabel_iter)
        except Exception as err:
            unlabel_iter = iter(loader_ULtrain)
            target_data, _, tar_idx = next(unlabel_iter)
        if len(target_data)!= unlabel_size or len(source_data) != batch_size:
            print('continue')
            continue
        if len(target_data)!= len(tar_idx) or len(source_data)!= len(src_idx):
            print(len(target_data))
            print(len(tar_idx))
            print('Error')
            break      
        source_data, source_label = source_data.float().to(device), source_label.float().to(device)
        target_data,tar_idx = target_data.float().to(device),tar_idx.long()
        optimizer.zero_grad()
        
        with torch.no_grad():
            ## ECGAugment
            aug_source_data_weak=tar_augmentation(source_data,'Weak',device)
            aug_target_data_weak=tar_augmentation(target_data,'Weak',device)
            aug_target_data_strong=tar_augmentation(target_data,'Strong',device)
            model_teacher.eval()
            model.eval()
            
            ## update the banks on the fly
            features_tar = model_teacher.bottleneck(model_teacher.feature_extraction(aug_target_data_weak))
            pred_tar=model_teacher.classifier(features_tar).sigmoid()
            fea_bank_UL[tar_idx] = F.normalize(features_tar).cpu().detach().clone()
            score_bank_UL[tar_idx] = pred_tar.detach().clone()
            
            ## search K-nearest neighbors for the weak-augented unlabeled data
            features_tar_stud = model.bottleneck(model.feature_extraction(aug_target_data_weak))
            fea_tar_norm=F.normalize(features_tar_stud).cpu().detach().clone()
            distance = fea_tar_norm @ fea_bank_UL.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=K)
            
            ## pseudo-label generation: compute the pseudo-labels for the unlabeled samples using a soft-voting method
            score_near_tar = torch.mean(score_bank_UL[idx_near],axis=1)  # batch x K x C
            
            ## pseudo-label refinement: compute the agreement weight
            agreement_level = torch.sum(score_bank_UL[idx_near],axis=1)
            weight=torch.abs(agreement_level*(2/K)-1)
            
        model.train()
        inputs = torch.cat((aug_source_data_weak, aug_target_data_weak, aug_target_data_strong))
        logits,_ = model(inputs.float())
        logits=logits.sigmoid()
        logits_x_lb = logits[:batch_size]
        _, logits_x_ulb_s = logits[batch_size:].chunk(2)
        
        ## compute the supervised loss using labeled samples
        sup_loss = criterion(logits_x_lb, source_label)
        
        ## label correlation alignment
        relationship_mat_UL=similarity_metrics(logits[batch_size:], args.similarity)
        relationship_mat_UL=torch.clip(relationship_mat_UL,0,1)
        relationship_loss=torch.norm(relationship_mat_UL-relationship_mat_L,'fro')
        
        ## compute the unsupervised loss using unlabeled samples
        neighbor_loss = F.binary_cross_entropy(logits_x_ulb_s,score_near_tar.float(),reduction='none')
            
        ## final loss
        if args.neighbor_weight:
            loss = sup_loss+args.Semi_loss_weight*torch.mean(weight*neighbor_loss)+args.relationship_weight*relationship_loss
        else:
            loss = sup_loss+args.Semi_loss_weight*torch.mean(neighbor_loss)+args.relationship_weight*relationship_loss
            
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        with torch.no_grad():
            EMA_model.update(model)
    
    ## compute the best classcification thresholds using validation data 
    valid_f1_list=[]   
    for epoch in range(len(valid_result_list)):
        valid_f1_list.append(valid_result_list[epoch][args.metrics])
    threshold=valid_result_list[np.argmax(valid_f1_list)]['threshold']
    os.chdir('/home/coche/RushuangZhou_Phdstudent/Cinc2021data/result')
    if args.teacher_pretrain:
        model.load_state_dict(torch.load(dataset_name +semi_config+'_'+str(int(10*args.Semi_loss_weight))+'_'+str(int(10*args.relationship_weight))+'_'+ema_flag+'_checkpoint_'+str(args.seed)+'.pkl'))
    else:
        model.load_state_dict(torch.load(dataset_name +semi_config+'_'+str(int(10*args.Semi_loss_weight))+'_'+str(int(10*args.relationship_weight))+'_'+ema_flag+'_checkpoint_'+str(args.seed)+'_pre.pkl'))
    
    ## testing 
    print('test')
    retrain_test_result = validate(model, loader_test, device,iftest=True,threshold=threshold,args=args)
    return model,retrain_test_result,train_result_list, ULtrain_result_list,valid_result_list



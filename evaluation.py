# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:39:27 2022

@author: COCHE User
"""

from sklearn.metrics import roc_auc_score, hamming_loss,label_ranking_loss,accuracy_score,precision_recall_curve
from sklearn.metrics import coverage_error
import numpy as np

def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=True):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
    f_beta_each_class = []
    g_beta_each_class = []
    for classi in range(y_true.shape[1]):
        y_truei, y_predi = y_true[:,classi], y_pred[:,classi]
        TP, FP, TN, FN = 0.,0.,0.,0.
        for i in range(len(y_predi)):
            sample_weight = sample_weights[i]
            if y_truei[i]==y_predi[i]==1:
                TP += 1./sample_weight
            if ((y_predi[i]==1) and (y_truei[i]!=y_predi[i])):
                FP += 1./sample_weight
            if y_truei[i]==y_predi[i]==0:
                TN += 1./sample_weight
            if ((y_predi[i]==0) and (y_truei[i]!=y_predi[i])):
                FN += 1./sample_weight
        f_beta_i = ((1+beta1**2)*TP)/((1+beta1**2)*TP + FP + (beta1**2)*FN)
        g_beta_i = (TP)/(TP+FP+beta2*FN)
        f_beta += f_beta_i
        g_beta += g_beta_i
        f_beta_each_class.append(f_beta_i)
        g_beta_each_class.append(g_beta_i)
    return f_beta/y_true.shape[1], g_beta/y_true.shape[1],f_beta_each_class,g_beta_each_class

def average_precision(output, target):
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i

def mAP(targs, preds):
    """Returns the model's average precision for each class
    Return:
        ap (FloatTensor): 1xK tensor, with avg precision for each class k
    """

    if np.size(preds) == 0:
        return 0
    ap = np.zeros((preds.shape[1]))
    # compute average precision for each class
    for k in range(preds.shape[1]):
        # sort scores
        scores = preds[:, k]
        targets = targs[:, k]
        # compute average precision
        ap[k] = average_precision(scores, targets)
    return 100 * ap.mean(), 100*ap

def evaluation(label,predict,thres=0.5):
    ## logit-based
    auc,auc_each_class=roc_auc_score(label,predict, average='macro'),roc_auc_score(label, predict, average=None)
    rankingloss=label_ranking_loss(label,predict)
    Coverage=coverage_error(label,predict)
    Map_value,map_each_class=mAP(label,predict)
    ## one-hot-based
    for i in range(predict.shape[1]):
        predict[:, i] = (predict[:, i] > thres[i]) + 0.0
    # predict=(predict>thres)+0
    hammingloss=hamming_loss(label,predict)
    acc=accuracy_score(label,predict)
    F1score_b,Gscore_b,f_beta_each_class,g_beta_each_class=challenge_metrics(label,predict)   
    F1score,_,f_each_class,_=challenge_metrics(label,predict,beta1=1,beta2=1)   
    performance_table={'auc':auc,'ranking':rankingloss,'hamming':hammingloss,'acc':acc,'F1score_b':F1score_b,
                       'Gscore_b':Gscore_b,'Map_value':Map_value,'Coverage':Coverage,
                       'auc_class':auc_each_class,'map_class': map_each_class,
                       'F1score_b_class':f_beta_each_class,'Gscore_b_class':g_beta_each_class,
                       'F1score':F1score,'F1score_class':f_each_class}
    return performance_table

def print_result(loss,label,predict,datatype,thres=0.5*np.ones(5)):
    performance_table=evaluation(label,predict,thres=thres)
    print(datatype+'_loss: '+str(loss))
    print(datatype+'_auc: '+str(performance_table['auc']))
    print(datatype+'_ranking: '+str(performance_table['ranking']))
    print(datatype+'_hamming: '+str(performance_table['hamming']))
    print(datatype+'_F1score_b: '+str(performance_table['F1score_b']))
    print(datatype+'_F1score: '+str(performance_table['F1score']))
    print(datatype+'_Gscore_b: '+str(performance_table['Gscore_b']))
    print(datatype + '_MAPvalue: ' + str(performance_table['Map_value']))
    print(datatype + '_Coverage: ' + str(performance_table['Coverage']))
    performance_table.update({'yloss':loss})
    performance_table.update({'threshold': thres})
    return performance_table

def find_thresholds(label,predict,beta=2):
    N = label.shape[1]
    f1prcT = np.zeros((N,))
    for j in range(N):
        prc, rec, thr = precision_recall_curve(y_true=label[:, j], probas_pred=predict[:, j])
        fscore = (1+beta**2) * prc * rec / ((beta**2)*prc + rec)
        idx = np.nanargmax(fscore)
        f1prcT[j] = thr[idx]
    return f1prcT


    
    
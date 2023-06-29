# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:39:27 2022

@author: COCHE User
"""

from sklearn.metrics import roc_auc_score, hamming_loss,label_ranking_loss,accuracy_score,precision_recall_curve
from sklearn.neighbors import KernelDensity
from sklearn.metrics import coverage_error
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
def challenge_metrics(y_true, y_pred, beta1=2, beta2=2, class_weights=None, single=True):
    f_beta = 0
    g_beta = 0
    if single: # if evaluating single class in case of threshold-optimization
        sample_weights = np.ones(y_true.sum(axis=1).shape)
    else:
        sample_weights = y_true.sum(axis=1)
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
    return f_beta/y_true.shape[1], g_beta/y_true.shape[1]
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
    return 100 * ap.mean()
def evaluation(label,predict,thres=0.5):
    ## logit-based
    auc=roc_auc_score(label,predict, average='macro')
    rankingloss=label_ranking_loss(label,predict)
    Coverage=coverage_error(label,predict)
    Map_value=mAP(label,predict)
    ## one-hot-based
    for i in range(predict.shape[1]):
        predict[:, i] = (predict[:, i] > thres[i]) + 0.0
    # predict=(predict>thres)+0
    hammingloss=hamming_loss(label,predict)
    acc=accuracy_score(label,predict)
    F1score_b,Gscore_b=challenge_metrics(label,predict)   
    performance_table={'auc':auc,'ranking':rankingloss,'hamming':hammingloss,'acc':acc,'F1score_b':F1score_b,'Gscore_b':Gscore_b,'Map_value':Map_value,'Coverage':Coverage}
    return performance_table

def print_result(loss,label,predict,datatype,thres=0.5*np.ones(5)):
    performance_table=evaluation(label,predict,thres=thres)
    print(datatype+'_loss: '+str(loss))
    print(datatype+'_auc: '+str(performance_table['auc']))
    print(datatype+'_ranking: '+str(performance_table['ranking']))
    print(datatype+'_hamming: '+str(performance_table['hamming']))
    print(datatype+'_acc: '+str(performance_table['acc']))
    print(datatype+'_F1score_b: '+str(performance_table['F1score_b']))
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

def threshold_adaptation(predict,band_width=0.05,lam=0.95,initial_threshold=0.5*np.ones(5),initial_kde_threshold=0.5*np.ones(5)):
#    updated_threshold=[]
#    ## toy example
#    predict=np.ones((3000,5))
#    for i in range(len(intial_threshold)):
#        predict[:,i]=np.random.beta(0.5,0.5,3000)
#    file=np.load('baseline_noweightnorm_nonecknorm_ECGmatch_grid_search_cross_dataset_studentstrong_teacher_relationshipfro_withoutfix_weightneigh.npy',allow_pickle=True)[15][3]
#    label=np.load('ground_truth_WFDB_Ga_threshold_adaptation_super.npy')
#    predict=np.load('logit_WFDB_Ga_threshold_adaptation_super.npy')
#    predict=np.load('predict_WFDB_Ga_threshold_adaptation_super.npy')
#    intial_threshold=file['threshold']
    kde_threshold=np.zeros_like(initial_threshold)
    for i in range(len(initial_threshold)):
        score_distribution=predict[:,i]
        score_distribution=score_distribution.reshape(len(score_distribution),1)
        kde = KernelDensity(kernel='epanechnikov', bandwidth=band_width).fit(score_distribution)
        logit_samples=np.linspace(0,1,len(score_distribution))
#        logit_samples=np.linspace(max(score_distribution),min(score_distribution),len(score_distribution))
        logit_samples=logit_samples.reshape(len(score_distribution),1)
        log_density = kde.score_samples(logit_samples)
        plt.subplot(5,1,i+1)
        plt.plot(logit_samples,np.exp(log_density))
        peak_location=find_peaks(log_density)[0]
        if len(peak_location)<2:
            kde_threshold[i]=initial_threshold[i]
            continue
        cut_log_density=log_density[peak_location[0]:peak_location[-1]]
        cut_logit_samples=logit_samples[peak_location[0]:peak_location[-1]]
        kde_threshold[i]=cut_logit_samples[np.argmin(cut_log_density)]
    final_threshold=lam*initial_threshold+(1-lam)*kde_threshold
    final_threshold=initial_threshold*(kde_threshold/initial_kde_threshold)
#    threshold=find_thresholds(label,predict,beta=2)
#    predict=np.load('predict_WFDB_ChapmanShaoxing_threshold_adaptation_super.npy')
#    print_result(1,label,predict,'a',thres=final_threshold)
#    print_result(1,label,predict,'a',thres=intial_threshold)
#    print_result(1,label,predict,'a',thres=threshold)
    return final_threshold,kde_threshold

#        for iteration in range(max_iter):
#            density_current=kde.score_samples(intial_thres)
#        plt.plot(logit_samples,np.exp(log_density))
    #    label_distribution=label[:,i]
#        num=plt.hist(label_distribution)  num=plt.hist(score_distribution)

    
    
    
    
    
    
    
    
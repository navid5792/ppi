#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 20:25:34 2018

@author: Jumayel
"""
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
import pickle
true, pred, _ = pickle.load(open('./true pred/true_pred_IEPA.pkl', 'rb'))
prec, reca, fscore, sup = precision_recall_fscore_support(true, pred, average='macro')
print("precision: ", prec, " recall: ", reca, " fscore: ", fscore)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)
   

tp, fp, tn, fn = perf_measure(true, pred)
prec_pos = tp/(tp+fp)
prec_neg = tn/(tn+fn)

rec_pos = tp/(tp+fn)
rec_neg = tn/(tn+fp)

fscore_pos = 2*prec_pos*rec_pos/(prec_pos + rec_pos)
fscore_neg = 2*prec_neg*rec_neg/(prec_neg + rec_neg)

print('Precision (Avg.): ', (prec_pos + prec_neg)/2)
print('Recall (Avg.): ', (rec_pos + rec_neg)/2)
print('Fscore (Avg.): ', (fscore_pos + fscore_neg)/2)

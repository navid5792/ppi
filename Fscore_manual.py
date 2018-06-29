#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 20:25:34 2018

@author: bob
"""
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
import pickle
true, pred = pickle.load(open('./true pred/true_pred_HPRD50.pkl', 'rb'))
prec, reca, fscore, sup = precision_recall_fscore_support(true, pred, average='macro')
print("precision: ", prec, " recall: ", reca, " fscore: ", fscore)

def performance_metrics(label, labels, actual_labels, predicted_labels):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    other = list(set(labels) - set([label]))
    for i in range(len(predicted_labels)):
        actual = actual_labels[i]
        predicted = predicted_labels[i]
        if actual == label and predicted == label:
            true_positive += 1
        if actual in other and predicted == label:
            false_positive += 1
        if actual == label and predicted in other:
            false_negative += 1

    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    fscore = 2*precision*recall/(precision + recall)

    return precision, recall, fscore  

precision = []
recall = []
fscore = []   
    
for label in [0,1]:
   pre , re, fs = performance_metrics(label,[0,1],true,pred)
   precision.append(pre)
   recall.append(re)
   fscore.append(fs)
print(precision, recall, fscore)
import numpy as np
print(np.mean(precision))
print(np.mean(recall))
print(np.mean(fscore))
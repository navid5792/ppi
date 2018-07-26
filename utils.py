#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 22:55:48 2018

@author: bob
"""

import pickle
from nltk import pos_tag
import numpy  as np
from tree import *
from collections import Counter 
from copy import deepcopy
data = []
labels = []
parents = []
file = "HPRD50"

def dataset_stat():
    global data
    global labels
    global parents
    with open("PPI Corpora/%s/sentences.toks" % file, "r") as f:
        data = f.readlines()
        
    with open("PPI Corpora/%s/labels.txt" % file, "r") as f:
        labels  = f.readlines()
    
    with open("PPI Corpora/%s/sentences.cparents" % file, "r") as f:
        parents  = f.readlines()
    Z = 0
    O = 0
    Z_index = []
    O_index = []
    for i in range(len(labels)):
        labels[i] = int(labels[i].strip())
        if(labels[i] == 0):
            Z += 1
            Z_index.append(i)
        else:
            O += 1
            O_index.append(i)
    return len(Z_index), len(O_index)

def ten_fold():
    from sklearn.model_selection import KFold
    z_l, o_l = dataset_stat() # 4784 991
    data_ones = data[:o_l]
    data_zeros = data[o_l:o_l + z_l] 
    np.random.shuffle(data_zeros)
    np.random.shuffle(data_ones)
    ''' 10 fold for one label'''   
    kf = KFold(n_splits=10)
    kf.get_n_splits(data_ones)  
    train_indexes_1 =[]
    test_indexes_1 = []
    for train_index, test_index in kf.split(data_ones):
        train_indexes_1.append(train_index)
        test_indexes_1.append(test_index)
        
    ''' 10 fold for zero label'''   
    kf = KFold(n_splits=10)
    kf.get_n_splits(data_zeros)  
    train_indexes_0 =[]
    test_indexes_0 = []
    for train_index, test_index in kf.split(data_zeros):
        train_indexes_0.append([x+o_l for x in list(train_index)])
        test_indexes_0.append([x+o_l for x in list(test_index)])

    X = []
    Y = []
    for i in range(len(train_indexes_0)):
        x = list(train_indexes_0[i]) + list(train_indexes_1[i])
        np.random.shuffle(x)
        X.append (x)
        y = list(test_indexes_0[i]) + list(test_indexes_1[i])
        np.random.shuffle(y)
        Y.append (y)
    return X,Y

def create_vocab():
    dataset_stat()
    vocab = set()
    for i in range(len(data)):
        sen = data[i].strip().split()
        for j in range(len(sen)):
            vocab.add(sen[j])
    vocab = list(vocab)
    with open("./vocab and w2v/vocab.pkl", "wb") as f:
        pickle.dump(vocab,f)
    return vocab
            
def make_train_test():
    TR= []
    TE = []
    z_l, o_l = dataset_stat()
    tr, te = stratified_fold()
    
#    tr = []
#    te = []
#    for i in range(len(data)):
#        tr.append(i)
#        te.append(i)
#    tr = [tr]
#    te = [te]
    
    for j in range(len(tr)):
        train = []
        test = []
        for i in tr[j]:
            t_x = add_space_punc(data[i]).split()
            pos = [x[1] for x in pos_tag(t_x)]
            t_y = []
            tree = read_tree(parents[i].strip())
            temp = []
            t_y.append(int(labels[i]))
            temp.append(t_x)
            temp.append(pos)
            temp.append(t_y)
            temp.append(tree)
            train.append(temp)
            
        for i in te[j]:
            t_x = add_space_punc(data[i]).split()
            pos = [x[1] for x in pos_tag(t_x)]
            t_y = []
            tree = read_tree(parents[i].strip())
            temp = []
            t_y.append(int(labels[i]))
            temp.append(t_x)
            temp.append(pos)
            temp.append(t_y)
            temp.append(tree)
            test.append(temp)
        TR.append(train)
        TE.append(test) 
    return TR, TE           

def add_space_punc(s):
    import re
#    s = 'bla. bla? bla.bla! bla...'
#    s = re.sub('(?<! )(?=[.,!?()])|(?<=[.,!?()])(?! )', r' ', s)
    return s
 
def data_check():
    tr, te = make_train_test()
    c = 0
    for i in range(len(tr)):
        for j in range(len(tr[i])):
            if len(tr[i][j][0]) != len(tr[i][j][1]):
                c += 1
    print(c)
    maxlen = -1
    for i in range(len(data)):
        temp = data[i].strip().split()
        if len(temp) > maxlen:
            maxlen = len(temp)
    print(maxlen)

def stratified_fold():
    from sklearn.model_selection import StratifiedKFold
    from collections import Counter
    seed = 7
    np.random.seed(seed)
    dataset_stat()
    X = []
    Y = []
    for i in range(len(data)):
        temp = data[i].strip().split()
        X.append(temp)
        Y.append(labels[i])
    Y = np.array(Y)
    fold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = seed)
    x = []
    y = []
    for train, test in fold.split(X, Y):
        xx = list(train)
        np.random.shuffle(xx)
        x.append(xx)
        yy = list(test)
        np.random.shuffle(yy)
        y.append(yy)
        print("train--> ", Counter(Y[train]), "test-->", Counter(Y[test]))
#        print(len(Y[test]))
        print("----------")
    return x,y

from math import *
def chunkList(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out

def kfold():
    num_neg, num_pos= dataset_stat() 
    pos_indexes = []
    neg_indexes = []
    a = []
    b = []
    for i in range(num_pos):
        pos_indexes.append(i)
        
    for i in range(num_pos, num_pos+num_neg):
        neg_indexes.append(i)
        
    pos_splits = chunkList(pos_indexes, 10)
    neg_splits = chunkList(neg_indexes, 10)
    
    combined = list(zip(pos_splits, neg_splits))

    for i in range(10):
        temp_a = []
        temp_b = []
        
        pos_spc = []
        for j in range(10):
            if i != j:
                pos, neg = combined[j]
                pos_spc += pos
                temp_a += neg
            else:
                pos, neg = combined[j]
                temp_b += pos
                temp_b += neg
        
        toggle = True
        pos_mid = ceil(len(pos_spc)/2)
        for j in range(9):
            if toggle:
                temp_a += pos_spc[:pos_mid]
                toggle = False
            else:
                temp_a += pos_spc[pos_mid:]
                toggle = True
        np.random.shuffle(temp_a)
        np.random.shuffle(temp_b)
        a.append(temp_a)
        b.append(temp_b)
         
    return a,b  
 
def check_result():
    with open("true_pred.pkl", "rb") as f:
        true, pred = pickle.load(f)
    print("TRUE : ", Counter(true))
    print("PRED : ", Counter(pred))

#def try_CV():   
#    from sklearn.model_selection import KFold
#    #X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
#    #y = np.array([1, 2, 3, 4])
#    dataset_stat()
#    kf = KFold(n_splits=10)
#    kf.get_n_splits(data)
#    print(kf)  
#    for train_index, test_index in kf.split(data):
#        print("TRAIN:", len(train_index), "TEST:", len(test_index))
#    #       print("TRAIN:", train_index, len(train_index), "TEST:", test_index, len(test_index))
#    #       X_train, X_test = X[train_index], X[test_index]
#    #       y_train, y_test = y[train_index], y[test_index]
#        print(test_index)


#for i in range(o_l):
#    splitted = data[i].split()
#    t_x = []
#    t_y = []
#    temp = []
#    if i < o_l * (0.80):
#        t_y.append(int(labels[i].strip()))
#        temp.append(splitted)
#        temp.append(t_y)
#        train.append(temp)
#    else:
#        t_y.append(int(labels[i].strip()))
#        temp.append(splitted)
#        temp.append(t_y)
#        test.append(temp)
#
#for i in range(o_l, o_l + z_l):
#    splitted = data[i].split()
#    t_x = []
#    t_y = []
#    temp = []
#    if i < o_l + (z_l * 0.8):
#        t_y.append(int(labels[i].strip()))
#        temp.append(splitted)
#        temp.append(t_y)
#        train.append(temp)
#    else:
#        t_y.append(int(labels[i].strip()))
#        temp.append(splitted)
#        temp.append(t_y)
#        test.append(temp)   
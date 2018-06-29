#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 11:23:37 2018

@author: bob
"""

from tree import *
from tree_lstm import *

import pickle

trainn, testt = pickle.load(open("data/HPRD50.pkl", 'rb'))



for k in range(len(trainn)):
    tr = []
    te = []
    train = trainn[k]
    test = testt[k]
    
    for i in range(len(train)):
        tr.append(' '.join(train[i][0]))
        
    for i in range(len(test)):
        te.append(' '.join(test[i][0]))
        
    print(len(set(tr).intersection(set(te))))



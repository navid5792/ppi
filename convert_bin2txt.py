#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 19:41:50 2018

@author: bob
"""
import pickle
import torch
from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('PubMed-shuffle-win-2.bin', binary=True)
file = "IEPA"
with open("./vocab and w2v/vocab_%s.pkl" %file, "rb") as f:
    vocab = pickle.load(f)
w2v = dict()
found = 0
for i in range(len(vocab)):
    print(i)
    if vocab[i] in model:
        w2v[vocab[i]] = torch.FloatTensor(model[vocab[i]])
        found += 1
    else:
        w2v[vocab[i]] = torch.randn(200)
print(found)
with open("./vocab and w2v/pretrained_w2v_%s.pkl" %file, "wb") as f:
    pickle.dump(w2v,f)
print("dumped")

#model.save_word2vec_format('vector-win-2.txt', binary=False)
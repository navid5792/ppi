#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 19:47:31 2018

@author: ahmed
"""
import torch
import torch.nn as nn
#from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import pickle
import torch.nn.functional as F
from sklearn.metrics import f1_score,accuracy_score
from utils import dataset_stat, ten_fold, make_train_test, add_space_punc
from nltk import pos_tag
from torchviz import make_dot, make_dot_from_trace
from models import TreeAttention

USE_CUDA = True
batch_size = 30
no_epochs = 50
n_class = 2
hidden_dim = 400
embed_dim = 200
n_layer = 1
dropOut = 0.5
learning_rate = 0.015
lr_decay = 0.05
clip = 5
SGD = 0

with open("PPI Corpora/AIMed_/sentences.txt", "r") as f:
    data = f.readlines()
    
with open("PPI Corpora/AIMed_/labels.txt", "r") as f:
    labels  = f.readlines()

   
class vocabulary:
    def __init__(self):
        self.word2index = {'PAD' : 0}
        self.index2word = {0 : 'PAD'}
        self.n_words = 1
        self.pos2index = {'PAD' : 0}
        self.index2pos = {0 : 'PAD'}
        self.n_pos = 1
    def create_dict(self):
        set_data = set()
        set_pos = set()
        for i in range(len(data)):
            splitted = add_space_punc(data[i]).split()
            pos = [x[1] for x in pos_tag(splitted)]
            for j in range(len(splitted)):
                set_data.add(splitted[j])
                set_pos.add(pos[j])
        set_data = list(set_data)
        set_pos = list(set_pos)
        for i in range(len(set_data)):
            self.word2index[set_data[i]] = len(self.word2index)
            self.index2word[len(self.word2index)] = set_data[i]
            self.n_words += 1
        print("Vocabulary size: ", len(self.word2index))
        for i in range(len(set_pos)):
            self.pos2index[set_pos[i]] = len(self.pos2index)
            self.index2pos[len(self.pos2index)] = set_pos[i]
            self.n_pos += 1
        print("POS Vocabulary size: ", len(self.pos2index))   
    def dump_vocab(self):
        keys = list(self.word2index.keys())
        with open("vocab.pkl", "wb") as f:
            pickle.dump(keys,f)
vocab  = vocabulary()
vocab.create_dict()
vocab.dump_vocab()

with open("pretrained_w2v.pkl", "rb") as f:
    w2v = pickle.load(f)
keys = list(w2v.keys())
embed_tensor = torch.zeros(len(w2v), 200)
for i in range(len(keys)):
    embed_tensor[i] = w2v[keys[i]]
    
''' load dataset for 10 fold CV '''   
TR, TE = make_train_test()

def random_batch(index, pairs):
    ''' This will return  X tensor, Y tensor, length of X'''
    
    start = index * batch_size 
    end = index * batch_size + batch_size
    if index == num_of_batches:
        end = len(pairs)
    arr = np.arange(start, end)
    np.random.shuffle(arr)
    arr = list(arr)
    current_data = []
    current_label = []
    current_pos = []
    for k in arr:
        current_data.append(pairs[k][0])
        current_pos.append(pairs[k][1])
        current_label.append(pairs[k][2]) 
    length = []
    tensor_data = []
    tensor_pos = []
    for i in range(len(current_data)):
        temp = []
        temp_pos = []
        l = len(current_data[i]) 
        for j in range(l):
            temp.append(vocab.word2index[current_data[i][j]])
            temp_pos.append(vocab.pos2index[current_pos[i][j]])
        tensor_data.append(temp)
        tensor_pos.append(temp_pos)
        length.append(l)
    
    sorted_xy = sorted(zip(tensor_data,current_label, tensor_pos), key = lambda p: len(p[0]), reverse = True)
    tensor_data, tensor_label, tensor_pos = zip(*sorted_xy) 
    
    ''' calculate maximum length'''
    maxlen = max(length)
    for i in range(len(tensor_data)):
        l = len(tensor_data[i])
        for j in range(maxlen - l):
            tensor_data[i].append(vocab.word2index['PAD'])
            tensor_pos[i].append(vocab.pos2index['PAD'])

    ''' calculate actual length'''
    actual_length = []
    for i in range(len(tensor_data)):
        count = 0
        for j in range(len(tensor_data[i])):
            if tensor_data[i][j] == 0:
                count += 1
        actual_length.append(maxlen - count)
    
    ''' making tensor'''       
    tensor_data = torch.LongTensor(tensor_data)
    tensor_pos = torch.LongTensor(tensor_pos)
    tensor_label = torch.FloatTensor(tensor_label)
    if USE_CUDA:
        torch.cuda.set_device(0)
        tensor_data = tensor_data.cuda()
        tensor_label = tensor_label.cuda()
        tensor_pos = tensor_pos.cuda()
    #print(USE_CUDA,tensor_data.type())
    #asdf
    return tensor_data.transpose(0,1), tensor_label, tensor_pos.transpose(0,1), actual_length

class AIMED(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size):
        super(AIMED, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_pos = nn.Embedding(pos_vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, hidden_dim, num_layers = n_layer, dropout = dropOut, bidirectional = True )
        self.out = nn.Linear(hidden_dim * 4, n_class)
        self.drop = nn.Dropout(p = 0.1)
        self.sigmoid = nn.Sigmoid()
        self.w1 = nn.Parameter(torch.FloatTensor([1.0]))
        self.w2 = nn.Parameter(torch.FloatTensor([1.0]))
        self.a = None
        self.tree_attn = TreeAttention(hidden_dim*2, min_thres = -5, max_thres = 7, hard = False)
    
    def load_embeddings(self, tensor):
        self.embed.weight = nn.Parameter(tensor)
    
    def forward(self, x, y, p, lengths, hidden = None, hidden_pos = None, mode = "train"):
        emb = self.embed(x)
        emb = self.drop(emb)
        packed = pack_padded_sequence(emb, lengths)
        output, hidden = self.LSTM(packed, hidden)
        output, out_lengths = pad_packed_sequence(output)
        
#        print(output.size())
        tree_outputs = self.tree_attn(output, None, lengths).transpose(0,1)
#        print(tree_outputs.size())
        tree_forw = tree_outputs[-1,:,:hidden_dim].unsqueeze(0)
        tree_back = tree_outputs[0,:, hidden_dim:].unsqueeze(0)
        T = torch.cat([tree_forw, tree_back],0)
        T = torch.cat([T,hidden[0][:]])
        H = T[:].transpose(0,1).contiguous().view(emb.size(1),-1)
       
        output = output[:,:,:hidden_dim] + output[:,:,hidden_dim:]
        final_out = self.out(H)


#        final_out = self.out(output[-1,:,:])
#        final_out = self.sigmoid(final_out)
        '''pos section'''
#        emb_pos = self.embed_pos(p)
#        emb_pos = self.drop(emb_pos)
#        packed_pos = pack_padded_sequence(emb_pos, lengths)
#        output_pos, hidden_pos = self.LSTM(packed_pos, hidden_pos)
#        output_pos, out_lengths = pad_packed_sequence(output_pos)
#        output_pos = output_pos[:,:,:hidden_dim] + output_pos[:,:,hidden_dim:]
#        final_out_pos = self.out(hidden_pos[0][:].transpose(0,1).contiguous().view(emb.size(1),-1))
#        
#        final_out = self.w1 * final_out + self.w2 * final_out_pos
        
        final_out = F.softmax(final_out,1)
        
        
        a,b = final_out.topk(1)
        if USE_CUDA:
            pred = torch.cuda.FloatTensor(a)
        else:
            pred = torch.FloatTensor(a)
        if mode == "test":
            return b
        true = y
        #print(true.requires_grad)
        #print(pred.requires_grad)
        
        loss = crit(pred,true)
        self.a = output
        
        return loss
    def check_grad(self):
        print(self.a.grad)

best_f1 = -1
def evaluate(test):
    global best_f1
    model.train(False)
    num_correct = 0
    y_true = []
    y_pred = []
    zero = 0
    one = 0
    for i in range(len(test)):
        test_data = test[i][0]
        test_pos = test[i][1]
        test_label = test[i][2]
        length = [len(test_data)]
        test_x = []
        test_p = []
        for i in range(len(test_data)):
            test_x.append(vocab.word2index[test_data[i]])
            test_p.append(vocab.pos2index[test_pos[i]])
        test_data = torch.LongTensor(test_x).unsqueeze(1)
        test_pos = torch.LongTensor(test_p).unsqueeze(1)
        test_label = torch.FloatTensor(test_label)
        if USE_CUDA:
            test_data = test_data.cuda()
            test_label = test_label.cuda()
            test_pos = test_pos.cuda()
        out = model(test_data, test_label, test_pos, length, mode = "test")
        if int(out.data.cpu().numpy()) == int(test_label.cpu().numpy()):
            num_correct += 1
        y_true.append(int(test_label.cpu().numpy()))
        y_pred.append(int(out.data.cpu().numpy()))
        if int(test_label.cpu().numpy()) == 0:
            zero += 1
        else:
            one += 1
    print("ZZZZ", zero, "Onweeeee", one)
#    acc = num_correct/len(test)
    f1 = f1_score(y_true, y_pred, average="micro")
    acc = accuracy_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        with open("true_pred.pkl", "wb") as f:
            pickle.dump([y_true, y_pred],f)            
    model.train(True)
    return f1

#model = AIMED(len(vocab.word2index), len(vocab.pos2index))
#model.load_embeddings(embed_tensor)
#if SGD == 1:
#    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
#else:
#    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
#
#crit = nn.BCELoss()
#if USE_CUDA:
#    model.cuda()
#print(model)


total_acu = []
print(len(TR))
LR = torch.FloatTensor([0.1])
for k in range(len(TR)):
    model = AIMED(len(vocab.word2index), len(vocab.pos2index))
    model.load_embeddings(embed_tensor)
    if SGD == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    crit = nn.BCELoss()
    if USE_CUDA:
        model.cuda()
    print(model)

    train = TR[k]
    test = TE[k]
    num_of_batches = int(len(train)/ batch_size)
    epoch = 0
    ac_loss = []
    while epoch < no_epochs:
        cur_batch = 0
        total_loss = 0
        model.train(True)
        while cur_batch < num_of_batches:
            optimizer.zero_grad()
#            try: 
#                model.w1.grad.data.zero_()
#                model.w2.grad.data.zero_()
#                print("Inside first try")
#            except Exception as e:
#                print("first exeption ", e)
#                pass
            X, Y, P, lengths = random_batch(cur_batch, train)
            zero = 0
            one = 0
            for i in range(Y.size(0)):
                if int(Y[i].data) == 0:
                    zero +=1
                else:
                    one +=1
            
            print("Zero:",zero,"One:",one)
            import time
#            time.sleep(0)
            #asdf
            loss = model(X, Y, P, lengths)
            #print(loss)
#            make_dot(loss.mean(), params=dict(model.named_parameters()))
            print("aimed_T FOLD ", k," epoch ", epoch, "Batch: ", cur_batch, "   ", float(loss.data.cpu().numpy()))
#            make_dot(loss.mean(), params = dict(model.named_parameters()))

            loss.backward()
            
            optimizer.step()
            
#            try:
#                model.w1 = model.w1.sub(LR * model.w1.grad)
#                model.w2 = model.w2.sub(LR * model.w2.grad)
#                print("Inside Try")
#            except Exception as e:
#                print("Exception", e)
#                pass
            
            total_loss += float(loss.data.cpu().numpy())
            cur_batch += 1 
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)

#            print(model.w1.grad,model.w1.data)
            
        
        if SGD == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / (1 + epoch * lr_decay)
       
        acu = evaluate(test)
        mean_loss = total_loss/num_of_batches
        print("Attn aimed Mean Loss: ", mean_loss, " Accuracy: ", acu, "FOLD ", k ,"completed", "w1 ", float(model.w1.data.cpu().numpy()), " and w2 ", float(model.w2.data.cpu().numpy()))
        ac_loss.append((acu, mean_loss))
        with open("result.pkl", "wb") as f:
            pickle.dump(ac_loss,f)
        epoch += 1
    total_acu.append(acu)

print(total_acu, sum(total_acu) / 10)
        
     
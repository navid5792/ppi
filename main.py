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
#from torchviz import make_dot, make_dot_from_trace
from tree import *
from tree_lstm import *
from copy import deepcopy
from PPI_class import PPI, PPI_attn
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support

USE_CUDA = False
batch_size = 10
no_epochs = 30
n_class = 2
hidden_dim = 300
embed_dim = 200
n_layer = 1
dropOut = 0.5
lr_decay = 0.05
clip = 5
SGD = 0
file = "HPRD50"
logfile = "fscore_%s.txt" %file

if SGD == 1:
    learning_rate = 0.015
else:
    learning_rate = 0.001

with open("PPI Corpora/%s/sentences.toks" %file, "r") as f:
    data = f.readlines()
    
#with open("PPI Corpora/%s/labels.txt" %file, "r") as f:
#    labels  = f.readlines()

   
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
        with open("./vocab and w2v/vocab_%s.pkl" %file, "wb") as f:
            pickle.dump(keys,f)
vocab  = vocabulary()
vocab.create_dict()
vocab.dump_vocab()

embed_tensor = torch.zeros([])
def load_pretrained_vector():
    global embed_tensor
    with open("./vocab and w2v/pretrained_w2v_%s.pkl" %file, "rb") as f:
        w2v = pickle.load(f)
    tensor = torch.zeros(len(w2v), 200)
    keys = list(w2v.keys())
    for i in range(len(keys)):
        tensor[i] = w2v[keys[i]]
    embed_tensor = deepcopy(tensor)

TR = []
TE = []
def dump_load_data(mode):
    global TR
    global TE
    if mode == 1:
        ''' load dataset for 10 fold CV '''
        TR, TE = make_train_test()
        with open ("./data/%s.pkl" %file, "wb") as f:
            pickle.dump([TR, TE], f)
    else:
        with open ("./data/%s.pkl" %file, "rb") as f:
            TR, TE = pickle.load(f)
        print("data loaded")

load_pretrained_vector()
dump_load_data(0)   


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
    current_tree = []
    for k in arr:
        current_data.append(pairs[k][0])
        current_pos.append(pairs[k][1])
        current_label.append(pairs[k][2]) 
        current_tree.append(pairs[k][3])
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
    
    sorted_xy = sorted(zip(tensor_data,current_label, tensor_pos, current_tree), key = lambda p: len(p[0]), reverse = True)
    tensor_data, tensor_label, tensor_pos, current_tree = zip(*sorted_xy) 
    
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
    return tensor_data.transpose(0,1), tensor_label, tensor_pos.transpose(0,1), current_tree, actual_length

class AIMED(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size):
        super(AIMED, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.embed_pos = nn.Embedding(pos_vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, hidden_dim, num_layers = n_layer, dropout = dropOut, bidirectional = True )
        self.out = nn.Linear(hidden_dim * 1, n_class)
        self.drop = nn.Dropout(p = 0.1)
        self.sigmoid = nn.Sigmoid()
        self.w1 = nn.Parameter(torch.FloatTensor([1.0]))
        self.w2 = nn.Parameter(torch.FloatTensor([1.0]))
        self.childsumtreelstm = ChildSumTreeLSTM(embed_dim, hidden_dim)
    
    def load_embeddings(self, tensor):
        self.embed.weight = nn.Parameter(tensor)
    
    def forward(self, x, y, p, t, lengths, hidden = None, hidden_pos = None, mode = "train"):
        emb = self.embed(x)
        emb = self.drop(emb)
        
        ''' dependency tree embedding '''
        tree_state = torch.zeros(x.size(1), hidden_dim)
        tree_hidden = torch.zeros(x.size(1), hidden_dim)
        for k in range (len(t)):
            t_state, t_hidden = self.childsumtreelstm(t[k], emb.transpose(0,1)[k].squeeze(0)) 
            tree_state[k] = t_state
            tree_hidden[k] = t_hidden
        tree_out = torch.cat([tree_state, tree_hidden], 1)
                   
        ''' word section '''
        packed = pack_padded_sequence(emb, lengths)
        output, hidden = self.LSTM(packed, hidden)
        output, out_lengths = pad_packed_sequence(output)
        output = output[:,:,:hidden_dim] + output[:,:,hidden_dim:]

#        final_out = self.out(output[-1,:,:])
#        final_out = self.sigmoid(final_out)
        '''pos section'''
        emb_pos = self.embed_pos(p)
        emb_pos = self.drop(emb_pos)
        packed_pos = pack_padded_sequence(emb_pos, lengths)
        output_pos, hidden_pos = self.LSTM(packed_pos, hidden_pos)
        output_pos, out_lengths = pad_packed_sequence(output_pos)
        output_pos = output_pos[:,:,:hidden_dim] + output_pos[:,:,hidden_dim:]
        
#        ''' dependency POS '''
#        tree_state_pos = torch.zeros(x.size(1), hidden_dim)
#        tree_hidden_pos = torch.zeros(x.size(1), hidden_dim)
#        for k in range (len(t)):
#            t_state_pos, t_hidden_pos = self.childsumtreelstm(t[k], emb_pos.transpose(0,1)[k].squeeze(0)) 
#            tree_state_pos[k] = t_state_pos
#            tree_hidden_pos[k] = t_hidden_pos
#        tree_out_pos = torch.cat([tree_state_pos, tree_hidden_pos], 1)
        
        ''' pick the last hidden '''
        final_hidden = hidden[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)
        final_hidden_pos = hidden_pos[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)
        
        ''' apply W's '''
#        final_out = self.w1 * final_hidden + self.w2 * final_hidden_pos
        
#        tree_out_ = self.w1 * tree_hidden + self.w2 * tree_hidden_pos
        
        ''' Final FC and argmax '''
        if USE_CUDA:
            tree_out = tree_out.cuda()
            tree_state = tree_state.cuda()
            tree_hidden = tree_hidden.cuda()
#            tree_out_pos = tree_out_pos.cuda()
#            tree_state_pos = tree_state_pos.cuda()
#            tree_hidden_pos = tree_hidden_pos.cuda()
#            
#        final_out = self.out(tree_out)
        final_out = self.out(torch.cat([tree_hidden],1))
        final_out = F.logsoftmax(final_out,1)
        a,b = final_out.topk(1)

        if USE_CUDA:
            pred = torch.cuda.FloatTensor(a)
        else:
            pred = torch.FloatTensor(a)
        if mode == "test":
            return b
        true = y
        #print(final_out.cpu().type(),true.cpu().long().transpose(0,1).type())
#        loss = crit(final_out.cpu(),true.cpu().long().squeeze(1))
        loss = crit(pred, true)
        return loss


best_f1 = -1
def evaluate(test):
    global best_f1
    model.train(False)
    num_correct = 0
    y_true = []
    y_pred = []
    true_sentence = []
    zero = 0
    one = 0
    for k in range(len(test)):
        test_data = test[k][0]
        test_pos = test[k][1]
        test_label = test[k][2]
        test_tree = [test[k][3]]
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
        out = model(test_data, test_label, test_pos, test_tree, length, mode = "test")
        if int(out.data.cpu().numpy()) == int(test_label.cpu().numpy()):
            num_correct += 1
        y_true.append(int(test_label.cpu().numpy()))
        y_pred.append(int(out.data.cpu().numpy()))
        true_sentence.append(" ".join(test[k][0]))
        if int(test_label.cpu().numpy()) == 0:
            zero += 1
        else:
            one += 1
    print("ZZZZ", zero, "Oneeeee", one)
#    acc = num_correct/len(test)
    #f1 = f1_score(y_true, y_pred, average="macro")
    
    prec, reca, f1, sup = precision_recall_fscore_support(y_true, y_pred, average='macro')
    
    acc = accuracy_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        with open("./true pred/true_pred_%s.pkl" %file, "wb") as f:
            pickle.dump([y_true, y_pred,true_sentence],f)            
    model.train(True)
    return prec, reca, f1


total_acu = []
print(len(TR))
LR = torch.FloatTensor([0.1])
fscoreList = open(logfile, 'w')
fscoreList.close()

def weight_init(): 
    for x in model.named_parameters():
        if len(x[1].size()) > 1:
            print(x[0], " initialized") 
            torch.nn.init.xavier_uniform(x[1])
        else:
            print(x[0], " initialized")
            x[1].data = torch.randn(x[1].size())
        if USE_CUDA:
            model.cuda()



max_f1_scores = []
max_precision_scores = []
max_recall_scores = []


for k in range((len(TR))):
    
    ''' model initialize '''
    crit = nn.BCELoss()
    model = PPI_attn(len(vocab.word2index), len(vocab.pos2index), hidden_dim, embed_dim, n_class, USE_CUDA, crit)
    model.load_embeddings(embed_tensor)
    if SGD == 1:
        optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = 0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-5)
    
    #    crit = nn.NLLLoss(torch.tensor([0.7, 0.3]))
    
    if USE_CUDA:
        model.cuda()
    print(model)  
   
    train = TR[k]
    test = TE[k]
    num_of_batches = int(len(train)/ batch_size)
    epoch = 0
    ac_loss = []
    max_precision = 0
    max_recall = 0
    max_f1 = 0
    max_f1_epoch = 0
    while epoch < no_epochs:
        fscoreList = open(logfile, 'a')
        fscoreList.write('Fold ' + str(k + 1) + ' ')
        fscoreList.write('Epoch ' + str(epoch) + ': ')
        cur_batch = 0
        total_loss = 0
        model.train(True)
 
        while cur_batch < num_of_batches:
            optimizer.zero_grad()
            X, Y, P, T,  lengths = random_batch(cur_batch, train)
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
#            asdf
            loss = model(X, Y, P, T,  lengths)
#            print(loss)
#            make_dot(loss.mean(), params=dict(model.named_parameters()))

            print("FOLD ", k," epoch ", epoch, "Batch: ", cur_batch, "   ", float(loss.data.cpu().numpy()))
#            make_dot(loss.mean(), params = dict(model.named_parameters()))
            loss.backward()
            
            optimizer.step()
            
            total_loss += float(loss.data.cpu().numpy())
            cur_batch += 1 
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
#            print(model.w1.grad,model.w1.data)

        np.random.shuffle(train)
        if SGD == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate / (1 + epoch * lr_decay)
       
        prec, recall, f1 = evaluate(test)
        
        fscoreList.write('Precision: ' + str(prec) + ', Recall: ' + str(recall) + ', Fscore: ' + str(f1) + '\n')
        fscoreList.close()
        if f1 > max_f1:
            max_precision = prec
            max_recall = recall
            max_f1 = f1
            max_f1_epoch = epoch
        
        mean_loss = total_loss/num_of_batches
        print("IEPA Mean Loss: ", mean_loss, " Accuracy: ", f1, "FOLD ", k ,"completed", "w1 ", float(model.w1.data.cpu().numpy()), " and w2 ", float(model.w2.data.cpu().numpy()))
        ac_loss.append((f1, mean_loss))
        with open("./results/result_%s.pkl" %file, "wb") as f:
            pickle.dump(ac_loss,f)
        epoch += 1
    fscoreList = open(logfile, 'a')
    max_precision_scores.append(max_precision)
    max_recall_scores.append(max_recall)
    max_f1_scores.append(max_f1)
    fscoreList.write('Max f1: ' + str(max_f1) + ' found in epoch #' + str(max_f1_epoch) + '\n')
    fscoreList.close()
    total_acu.append(f1)
    torch.cuda.empty_cache()
    #weight_init()
   
print('Max F1 Scores: ', max_f1_scores)
print('Mean Max Precision Scores: ', sum(max_precision_scores)/len(TR))
print('Mean Max Recall Scores: ', sum(max_recall_scores)/len(TR))
print('Mean Max F1 Scores: ', sum(max_f1_scores)/len(TR))
print(total_acu, sum(total_acu) / len(TR))
        
     
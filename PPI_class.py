#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 12:15:13 2018

@author: bob
"""
import torch
import torch.nn as nn
#from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
#from torchviz import make_dot, make_dot_from_trace
from tree import *
from tree_lstm import *
from copy import deepcopy
from models import TreeAttention
from new_utils import aeq
import math

class PPI(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size, hidden_dim, embed_dim, n_class, cuda, crit):
        super(PPI, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_class = n_class
        self.USE_CUDA = cuda
        self.crit = crit
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
#        self.embed_pos = nn.Embedding(pos_vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, hidden_dim, num_layers = 1, dropout = 0.5, bidirectional = True )
        self.out = nn.Linear(hidden_dim * 2, n_class)
        self.drop = nn.Dropout(p = 0.1)
        self.sigmoid = nn.Sigmoid()
        self.w1 = nn.Parameter(torch.FloatTensor([1.0]))
        self.w2 = nn.Parameter(torch.FloatTensor([1.0]))
#        self.childsumtreelstm = ChildSumTreeLSTM(self.embed_dim, self.hidden_dim)
        self.childsumtreelstm = BinaryTreeLSTM( self.USE_CUDA ,self.embed_dim, self.hidden_dim)
    
    def load_embeddings(self, tensor):
        self.embed.weight = nn.Parameter(tensor)
    
    def forward(self, x, y, p, t, lengths, hidden = None, hidden_pos = None, mode = "train"):
        emb = self.embed(x)
        emb = self.drop(emb)
        one_hot =[]
        for i in range(y.size(0)):
            if int(y[i]) == 0:
                one_hot.append([1.0, 0.0])
            else:
                one_hot.append([0.0, 1.0])
                           
        ''' word section '''
        packed = pack_padded_sequence(emb, lengths)
        output, hidden = self.LSTM(packed, hidden)
        output, out_lengths = pad_packed_sequence(output)
        output = output[:,:,:self.hidden_dim] + output[:,:,self.hidden_dim:]
        sent_vec = hidden[0][0]
        
#        ''' pick the last hidden '''
        final_hidden = hidden[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)
#        final_hidden_pos = hidden_pos[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)
#        
#        ''' apply W's '''
##        final_out = self.w1 * final_hidden + self.w2 * final_hidden_pos
#        
##        tree_out_ = self.w1 * tree_hidden + self.w2 * tree_hidden_pos
        
        ''' dependency tree embedding '''
        tree_state = torch.zeros(x.size(1), self.hidden_dim)
        tree_hidden = torch.zeros(x.size(1), self.hidden_dim)
        for k in range (len(t)):
            t_state, t_hidden = self.childsumtreelstm(t[k], emb.transpose(0,1)[k].squeeze(0), sent_vec[k]) 
            tree_state[k] = t_state
            tree_hidden[k] = t_hidden
        tree_out = torch.cat([tree_state, tree_hidden], 1)
        
        ''' Final FC and argmax '''
        if self.USE_CUDA:
            tree_out = tree_out.cuda()
            tree_state = tree_state.cuda()
            tree_hidden = tree_hidden.cuda()
#            tree_out_pos = tree_out_pos.cuda()
#            tree_state_pos = tree_state_pos.cuda()
#            tree_hidden_pos = tree_hidden_pos.cuda()
#            
#        final_out = self.out(tree_out)
        final_out = self.out(torch.cat([tree_out],1))
#        final_out = F.logsoftmax(final_out,1)
        final_out = self.sigmoid(final_out)
#        a,b = final_out.topk(1)

        if self.USE_CUDA:
            pred = torch.cuda.FloatTensor(final_out)
            true = torch.cuda.FloatTensor(one_hot)
        else:
            pred = torch.FloatTensor(final_out)
            true = torch.FloatTensor(one_hot)
        if mode == "test":
            a,b = final_out.topk(1)
            return b
#        true = y
        #print(final_out.cpu().type(),true.cpu().long().transpose(0,1).type())
#        loss = crit(final_out.cpu(),true.cpu().long().squeeze(1))
#        print(pred,pred.size(), true, true.size())
        loss = self.crit(pred, true)
        return loss

class PPI_attn(nn.Module):
    def __init__(self, vocab_size, pos_vocab_size, hidden_dim, embed_dim, n_class, cuda, crit):
        super(PPI_attn, self).__init__()
        self.childsum = True
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_class = n_class
        self.USE_CUDA = cuda
        self.crit = crit
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
#        self.embed_pos = nn.Embedding(pos_vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, hidden_dim, num_layers = 1, dropout = 0.5, bidirectional = True )
        self.out = nn.Linear(hidden_dim * 2, n_class)
        self.drop = nn.Dropout(p = 0.1)
        self.sigmoid = nn.Sigmoid()
        self.w1 = nn.Parameter(torch.FloatTensor([1.0]))
        self.w2 = nn.Parameter(torch.FloatTensor([1.0]))
        if self.childsum:
            self.childsumtreelstm = ChildSumTreeLSTM(self.embed_dim, self.hidden_dim, self.USE_CUDA )
        else:
            self.childsumtreelstm = BinaryTreeLSTM( self.USE_CUDA ,self.embed_dim, self.hidden_dim)
        self.tree_attn = TreeAttention(self.USE_CUDA, hidden_dim * 1, min_thres = -5, max_thres = 7, hard = False)
        self.linear_in = nn.Linear(hidden_dim, hidden_dim)
    
    def load_embeddings(self, tensor):
        self.embed.weight = nn.Parameter(tensor)
    
    def score(self, h_t, h_s):
        # Check input sizes
        src_batch, src_len, src_dim = h_s.size() # 10 87 300
        tgt_batch, tgt_len, tgt_dim = h_t.size() # 10 1 300
        aeq(src_batch, tgt_batch) # batch check
        aeq(src_dim, tgt_dim) # target check
        aeq(self.hidden_dim, src_dim) # dimension check

        h_t_ = h_t.view(tgt_batch*tgt_len, tgt_dim) # 10 300

        h_t_ = self.linear_in(h_t_) # 10 300
        
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim) # 10 1 300
        h_s_ = h_s.transpose(1, 2) # 10 300 87

        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len) 5 [1 32] x 5 [32 68] = 5 1 68
        return torch.bmm(h_t, h_s_) # 10 1 87
    
    def forward(self, x, y, p, t, lengths, hidden = None, hidden_pos = None, mode = "train"):
        emb = self.embed(x)
        emb = self.drop(emb)
        one_hot =[]
        for i in range(y.size(0)):
            if int(y[i]) == 0:
                one_hot.append([1.0, 0.0])
            else:
                one_hot.append([0.0, 1.0])
                   
        ''' word section '''
        packed = pack_padded_sequence(emb, lengths)
        output, hidden = self.LSTM(packed, hidden)

        sent_vec = hidden[0][0] # hidden[0][0].size() # 10 300
        output, out_lengths = pad_packed_sequence(output)
        output = output[:,:,:self.hidden_dim] + output[:,:,self.hidden_dim:]
        
#        ''' pick the last hidden '''
        final_hidden = hidden[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)

#        ''' apply W's '''
##        final_out = self.w1 * final_hidden + self.w2 * final_hidden_pos
        
        ''' dependency tree embedding '''
        if self.childsum == True:
            tree_state = torch.zeros(x.size(1), self.hidden_dim)
            tree_hidden = torch.zeros(x.size(1), self.hidden_dim)
            maxlen = emb.size(0)
            node_output = torch.zeros(emb.size(1), maxlen, self.hidden_dim).cuda() # 10 87 300 
    #        print(node_output.size(), output.size()) # 87 10 300 and 87 10 600
            for k in range (len(t)):
                node_out = output.transpose(0,1)[k]
                node_out = node_out [:,:self.hidden_dim]
                t_state, t_hidden = self.childsumtreelstm(t[k], emb.transpose(0,1)[k].squeeze(0), sent_vec[k]) 
    #            print(self.childsumtreelstm.words, emb.transpose(0,1)[k].size(), len(self.childsumtreelstm.words), len(self.childsumtreelstm.states))
                for x in self.childsumtreelstm.words:
                    node_out[x] = self.childsumtreelstm.states[x]
                tree_state[k] = t_state
                tree_hidden[k] = t_hidden
                node_output[k] = node_out
                self.childsumtreelstm.words = []
                self.childsumtreelstm.states = []
    
            tree_out = torch.cat([tree_state, tree_hidden], 1)
            node_output = node_output.transpose(0,1)
        else:
            tree_state = torch.zeros(x.size(1), self.hidden_dim)
            tree_hidden = torch.zeros(x.size(1), self.hidden_dim)
            for k in range (len(t)):
                t_state, t_hidden = self.childsumtreelstm(t[k], emb.transpose(0,1)[k].squeeze(0), sent_vec[k]) 
                tree_state[k] = t_state
                tree_hidden[k] = t_hidden
            tree_out = torch.cat([tree_state, tree_hidden], 1)
        
                
        ''' Final FC and argmax '''
        if self.USE_CUDA:
            tree_out = tree_out.cuda()
            tree_state = tree_state.cuda()
            tree_hidden = tree_hidden.cuda()

        ''' Tree attention '''
        tree_outputs = self.tree_attn(output, None, lengths) # 10 87 300
        align  = self.score(tree_hidden.unsqueeze(1), tree_outputs) # 10 1 87
        mask = torch.zeros(emb.size(1), emb.size(0))# 10 87
        if self.USE_CUDA:
            mask = mask.cuda()
        for g in range(len(lengths)):
            mask[g,lengths[g]:] = 1
        align.data.masked_fill_(mask.unsqueeze(1).byte(), -math.inf)
        align_vectors = F.softmax(align, dim=-1)
        context = torch.bmm(align_vectors, tree_outputs).squeeze(1) # 10 1 300
        
        
        final_out = self.out(torch.cat([tree_hidden, context],1))
#        final_out = F.logsoftmax(final_out,1)
        final_out = self.sigmoid(final_out)
#        a,b = final_out.topk(1)
        if self.USE_CUDA:
            pred = torch.cuda.FloatTensor(final_out)
            true = torch.cuda.FloatTensor(one_hot)
        else:
            pred = torch.FloatTensor(final_out)
            true = torch.FloatTensor(one_hot)
        if mode == "test":
            a,b = final_out.topk(1)
            return b

        loss = self.crit(pred, true)
        return loss

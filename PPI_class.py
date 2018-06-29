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
from torchviz import make_dot, make_dot_from_trace
from tree import *
from tree_lstm import *
from copy import deepcopy
from models import TreeAttention

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
        self.childsumtreelstm = ChildSumTreeLSTM(self.embed_dim, self.hidden_dim)
    
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
        
        ''' dependency tree embedding '''
        tree_state = torch.zeros(x.size(1), self.hidden_dim)
        tree_hidden = torch.zeros(x.size(1), self.hidden_dim)
        for k in range (len(t)):
            t_state, t_hidden = self.childsumtreelstm(t[k], emb.transpose(0,1)[k].squeeze(0)) 
            tree_state[k] = t_state
            tree_hidden[k] = t_hidden
        tree_out = torch.cat([tree_state, tree_hidden], 1)
                   
        ''' word section '''
        packed = pack_padded_sequence(emb, lengths)
        output, hidden = self.LSTM(packed, hidden)
        output, out_lengths = pad_packed_sequence(output)
        output = output[:,:,:self.hidden_dim] + output[:,:,self.hidden_dim:]
#
##        final_out = self.out(output[-1,:,:])
##        final_out = self.sigmoid(final_out)
#        '''pos section'''
#        emb_pos = self.embed_pos(p)
#        emb_pos = self.drop(emb_pos)
#        packed_pos = pack_padded_sequence(emb_pos, lengths)
#        output_pos, hidden_pos = self.LSTM(packed_pos, hidden_pos)
#        output_pos, out_lengths = pad_packed_sequence(output_pos)
#        output_pos = output_pos[:,:,:hidden_dim] + output_pos[:,:,hidden_dim:]
#        
##        ''' dependency POS '''
##        tree_state_pos = torch.zeros(x.size(1), hidden_dim)
##        tree_hidden_pos = torch.zeros(x.size(1), hidden_dim)
##        for k in range (len(t)):
##            t_state_pos, t_hidden_pos = self.childsumtreelstm(t[k], emb_pos.transpose(0,1)[k].squeeze(0)) 
##            tree_state_pos[k] = t_state_pos
##            tree_hidden_pos[k] = t_hidden_pos
##        tree_out_pos = torch.cat([tree_state_pos, tree_hidden_pos], 1)
#        
#        ''' pick the last hidden '''
        final_hidden = hidden[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)
#        final_hidden_pos = hidden_pos[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)
#        
#        ''' apply W's '''
##        final_out = self.w1 * final_hidden + self.w2 * final_hidden_pos
#        
##        tree_out_ = self.w1 * tree_hidden + self.w2 * tree_hidden_pos
        
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
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.n_class = n_class
        self.USE_CUDA = cuda
        self.crit = crit
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
#        self.embed_pos = nn.Embedding(pos_vocab_size, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, hidden_dim, num_layers = 1, dropout = 0.5, bidirectional = True )
        self.out = nn.Linear(hidden_dim * 3, n_class)
        self.drop = nn.Dropout(p = 0.1)
        self.sigmoid = nn.Sigmoid()
        self.w1 = nn.Parameter(torch.FloatTensor([1.0]))
        self.w2 = nn.Parameter(torch.FloatTensor([1.0]))
        self.childsumtreelstm = ChildSumTreeLSTM(self.embed_dim, self.hidden_dim)
        self.tree_attn = TreeAttention(self.USE_CUDA, hidden_dim * 2, min_thres = -5, max_thres = 7, hard = False)
    
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
        
        ''' dependency tree embedding '''
        tree_state = torch.zeros(x.size(1), self.hidden_dim)
        tree_hidden = torch.zeros(x.size(1), self.hidden_dim)
        for k in range (len(t)):
            t_state, t_hidden = self.childsumtreelstm(t[k], emb.transpose(0,1)[k].squeeze(0)) 
            tree_state[k] = t_state
            tree_hidden[k] = t_hidden
        tree_out = torch.cat([tree_state, tree_hidden], 1)
                   
        ''' word section '''
        packed = pack_padded_sequence(emb, lengths)
        output, hidden = self.LSTM(packed, hidden)
        output, out_lengths = pad_packed_sequence(output)
#        output = output[:,:,:self.hidden_dim] + output[:,:,self.hidden_dim:]
            
#        print(output.size())
        tree_outputs = self.tree_attn(output, None, lengths).transpose(0,1)

#        print(tree_outputs.size())
        tree_forw = tree_outputs[-1,:,:self.hidden_dim].unsqueeze(0)
        tree_back = tree_outputs[0,:, self.hidden_dim:].unsqueeze(0)

#
##        final_out = self.out(output[-1,:,:])
##        final_out = self.sigmoid(final_out)
#        '''pos section'''
#        emb_pos = self.embed_pos(p)
#        emb_pos = self.drop(emb_pos)
#        packed_pos = pack_padded_sequence(emb_pos, lengths)
#        output_pos, hidden_pos = self.LSTM(packed_pos, hidden_pos)
#        output_pos, out_lengths = pad_packed_sequence(output_pos)
#        output_pos = output_pos[:,:,:hidden_dim] + output_pos[:,:,hidden_dim:]
#        
##        ''' dependency POS '''
##        tree_state_pos = torch.zeros(x.size(1), hidden_dim)
##        tree_hidden_pos = torch.zeros(x.size(1), hidden_dim)
##        for k in range (len(t)):
##            t_state_pos, t_hidden_pos = self.childsumtreelstm(t[k], emb_pos.transpose(0,1)[k].squeeze(0)) 
##            tree_state_pos[k] = t_state_pos
##            tree_hidden_pos[k] = t_hidden_pos
##        tree_out_pos = torch.cat([tree_state_pos, tree_hidden_pos], 1)
#        
#        ''' pick the last hidden '''
        final_hidden = hidden[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)
#        final_hidden_pos = hidden_pos[0][:].transpose(0,1).contiguous().view(emb.size(1),-1)
#        
#        ''' apply W's '''
##        final_out = self.w1 * final_hidden + self.w2 * final_hidden_pos
#        
##        tree_out_ = self.w1 * tree_hidden + self.w2 * tree_hidden_pos
        
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
        final_out = self.out(torch.cat([tree_hidden, tree_forw.squeeze(0), tree_back.squeeze(0)],1))
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

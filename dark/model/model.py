#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture definitions for DARK models
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import LengthMask, TriangularCausalMask


class PositionalEncoding(torch.nn.Module):
    '''
    Positional Encoding from the original Transformer Vaswani et al (2017). 
    '''
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        pos_embedding =  self.pe[:, :x.size(1), :]
        pos_embedding = torch.repeat_interleave(pos_embedding, x.shape[0], dim=0)
        x =  torch.cat([x, pos_embedding], dim=2)
        return self.dropout(x)
    

class DARK(nn.Module):
    """
    Autoregressive DARK Model
    
    Note on Token Indices:
        0-19 AA's
        20-21 Start and Stop
    
    """
    def __init__(self, n_l=12,
                    n_h=12,
                    ff_d=768,
                    q_d=64,
                    v_d=64,
                    attention_type="full"):
        super().__init__()
        
        self.embed = nn.Embedding(22, ff_d//2)
        self.pos_encoder = PositionalEncoding(ff_d//2, max_len=101)
        
        # We use an encoder for simplicity sake but a causal mask must be passed
        # with a length mask for every forward pass
        self.AR = TransformerEncoderBuilder.from_kwargs(
                    n_layers=n_l,
                    n_heads=n_h,
                    feed_forward_dimensions=ff_d,
                    query_dimensions=q_d,
                    value_dimensions=v_d,
                    attention_type=attention_type 
                ).get()

         
        self.LM_head= nn.Sequential(
            nn.Linear(ff_d,128),
            nn.ReLU(),
            nn.Linear(128,22)
            )


    def forward(self,x, attn_mask=None, length_mask=None):
        x =self.embed(x)
        x =self.pos_encoder(x)
        x = self.AR(x,attn_mask,length_mask)
        x = self.LM_head(x)
        x = torch.log_softmax(x,2)
        return x
    
    
    def generate(self, num, device, L=100):
        x = torch.tensor([20]*num).long().unsqueeze(-1).to(device)
        for i in range(L):
            with torch.cuda.amp.autocast():
                new = self.forward(x,
                               attn_mask=TriangularCausalMask(x.shape[1],device=device),
                               length_mask=LengthMask(torch.tensor([x.shape[1]]*num),device=device))
            
            # sample
            x_i = torch.LongTensor([0]*num).to(device)
            for i in range(num):
                dist=new[i,-1].exp()[:20]
                x_i[i]=torch.multinomial(dist, 1)
            x = torch.cat((x,x_i.unsqueeze(-1)),-1)
        return x[:,1:]



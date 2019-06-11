import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import copy 

def copies(module: nn.Module, N):
    mods = [copy.deepcopy(module) for _ in range(N)]
    return nn.ModuleList(mods)

def to_one_hot(tensor: torch.Tensor, trg_vocab, batch_size):
    S = tensor.size(1) 
    l = torch.zeros(trg_vocab)
    fill = torch.zeros([batch_size, S, trg_vocab])

    for batch in range(batch_size):
        sent = tensor[batch]
        for j , w in enumerate(sent):
            word = copy.deepcopy(l)
            i = int(w) - 1
            word[i] = w.item()
            fill[batch][j] = word
    return fill

#def attention(query: torch.Tensor, 
#            key: torch.Tensor, 
#            value: torch.Tensor, 
#            dropout_rate  = None, mask = None):
#    """
#    dim query = (batch, embed_sent, sent_length)
#    dim key = (batch, embed, sent_length)
#    dim value = (batch, value_size, sen_length)
#    dim pdisro = (batch, embed)
#    dim attn = (batch, sent_length): how much of each word should we take form each sentence
#    """ 
#    norm = np.sqrt(query.size(-1))
#    key_t = key.transpose(-2, -1)
#    scores =  torch.matmul(query, key_t)
#    scores_norm = scores / norm
#    
#    if mask is not None:
#        scores_norm = scores_norm.masked_fill(mask == 0, -float('inf'))
#
#    attn_distro = F.softmax(scores_norm, dim = -1)
#
#    if dropout_rate is not None:
#        dropout = nn.Dropout(p=dropout_rate)
#        attn_distro = dropout(attn_distro) 
#    
#    attn =  torch.matmul(attn_distro, value)
#
#    return attn, attn_distro

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / np.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


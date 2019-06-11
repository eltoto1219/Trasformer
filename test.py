import matplotlib as plt
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
import torchtext
import random as r
import copy 

"""
I am going to make a transformer. 

"""
def to_one_hot(tensor: torch.Tensor, trg_vocab, batch_size):
    S = tensor.size(1) 
    l = torch.zeros(trg_vocab)
    fill = torch.zeros([batch_size, S, trg_vocab])

    for batch in range(N):
        sent = x[batch]
        for j , w in enumerate(sent):
            word = copy.deepcopy(l)
            i = int(w) - 1
            word[i] = w.item()
            fill[batch][j] = word
    return fill

if __name__ ==  "__main__":

    V = 11
    S = 10
    N = 1

    x = [r.randint(0, V) for _ in range(S)]  
    x = torch.Tensor([x])
    
    #main function for getting one hot vectors 

    print(to_one_hot(x, V, N))



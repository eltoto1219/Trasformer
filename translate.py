import os
from tqdm import tqdm
from model import Transformer, data_gen
import numpy as np
import time
import torchtext 
import numpy as np
import torch
import torch.nn as nn
from utils import to_one_hot
from dataloader import TRAIN_ITER, SRC, TRG
import model_components as mc

BATCH_SIZE = 1
DROPOUT = .0001
NHEADS = 8
EMBED_SIZE = 256 
VALUE_SIZE = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PATH = "/home/antonio/Desktop/projects/transformer/model_chkpnt_epoch_23.tar"

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           torch.autograd.Variable(ys), 
                           torch.autograd.Variable(mc.subsequent_mask(ys.size(1))
                                    .type_as(src.data)))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, 
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

VS  = len(SRC)
VT  = len(TRG)
pad = SRC.stoi['<pad>']

#loading state dict 
model = Transformer(VS, VT, EMBED_SIZE, VALUE_SIZE, NHEADS, pad_idx = pad,  drop_out = DROPOUT, device = device)
checkpoint = torch.load(PATH, map_location='cpu')
#model = checkpoint['model_state_dict']
#print(model)
#model.load_state_dict(checkpoint['model_state_dict'])
model.eval()




for i, batch in enumerate(TRAIN_ITER):
    batch = mc.Batch(batch.src[0].permute((1,0)), batch.trg[0].permute((1,0)), pad)
    #src_sent = batch.src.squeeze((0))
   # print(src_sent)
    #src_sent = [SRC.itos[x.item()] for x in src_sent]


    out = greedy_decode(model, batch.src, batch.src_mask, 
                        max_len=60, start_symbol=TRG.stoi["<s>"])
    print(out)
    print("Translation:", end="\t")
    for i in range(1, out.size(1)):
        sym = TRG.itos[out[0, i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print(batch.trg_y)
    print("Target:", end="\t")
    for i in range(1, batch.trg.size(1)):
        sym = TRG.itos[batch.trg[0,i]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    break

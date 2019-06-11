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

BATCH_SIZE = 32
DROPOUT = .0001
NHEADS = 8
EMBED_SIZE = 256 
VALUE_SIZE = 512
VOCAB_TEST = 1024
PATH = "/home/antonio/Desktop/check"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        #self.vocab = trg_vocab 
        
    def __call__(self, x, y, norm):
        B = x.size(0)
        x = self.generator(x)
        
        x_flat = x.contiguous().view(-1, x.size(-1)) 
        #y = to_one_hot(y, self.vocab, B)
        #y_flat = y.contiguous().view(-1, x.size(-1)) 
        y_flat = y.contiguous().view(-1) 
        loss = self.criterion(x_flat, y_flat) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        #print(loss.item())
        return loss.item() * norm

class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction = 'sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, torch.autograd.Variable(true_dist, requires_grad=False))

def run_epoch(data_iter, model, loss_compute, epoch_num):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    save_counter = 0
    for i, batch in enumerate(data_iter):
        save_counter +=1 
        out = model.forward(batch.src, batch.trg, 
                            None, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if save_counter == 50:
                save_counter = 0
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
            #and we will save the checkpoints here 

            p = os.path.join(PATH, "model_chkpnt_epoch_{}_{}.tar".format(epoch_num, i))
            torch.save({
                'model_state_dict': model.state_dict(),
                'loss': loss 
            }, p)

    return total_loss / total_tokens

if __name__ == "__main__":
    #DUMMY DATA FOR THE MODEK   
    #V = 11
    #criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    #model = Transformer(V, V, EMBED_SIZE, VALUE_SIZE, NHEADS, pad_idx = 0,  drop_out = DROPOUT, device = device)
    #model_opt = NoamOpt(EMBED_SIZE, 1, 10,
    #        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    ##hello
    #for epoch in range(10):
    #    epoch_num = epoch +1 
    #    loss = SimpleLossCompute(model.generator, criterion, model_opt)
    #    run_epoch(data_gen(V, 1000, 20), model, 
    #              SimpleLossCompute(model.generator, criterion,  model_opt), epoch_num)
    #    
    #    model.eval()
    #    print(run_epoch(data_gen(V, 30, 5), model, 
    #                    SimpleLossCompute(model.generator, criterion, None), epoch_num))

    ##REAL TRAIN BABY

    VS  = len(SRC)
    VT  = len(TRG)
    pad = SRC.stoi['<pad>']

    model = Transformer(VS, VT, EMBED_SIZE, VALUE_SIZE, NHEADS, pad_idx = pad,  drop_out = DROPOUT, device = device)
    model = model.to(device)
 
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    criterion = LabelSmoothing(size=VT, padding_idx=pad, smoothing=0.0)
    model_opt = NoamOpt(EMBED_SIZE, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(30):
        epochnum = epoch + 1
        model.train()
        run_epoch((mc.Batch(b.src[0].permute((1,0)), b.trg[0].permute((1,0)),pad) for b in tqdm(TRAIN_ITER)), 
                  model, 
                  SimpleLossCompute(model.generator, criterion, opt=model_opt), epochnum)
        model.eval()
        loss = run_epoch((mc.Batch(b.src[0].permute((1,0)), b.trg[0].permute((1,0)), pad) for b in TRAIN_ITER), model, 
            SimpleLossCompute(model.generator, criterion,  opt=None), epochnum)

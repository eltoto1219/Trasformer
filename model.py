import copy
import numpy as np 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from math import inf 
from model_components import Batch, PredictWord, LayerNorm, Embeddings, MultiHeadAttention, ResidualConnection, PositionalEncoding, FeedForward
from dataloader import TEST_BATCH, SRC, TRG

BATCH_SIZE = 32
DROPOUT = .0001
NHEADS = 8
EMBED_SIZE = 256
VALUE_SIZE = 512
VOCAB_TEST = 1024

class Transformer(nn.Module):
    def __init__(self, SV, TV,  embed_size, value_size, n_heads, pad_idx, drop_out, device):
        super(Transformer, self).__init__()
        self.src_vocab = SV 
        self.trg_vocab = TV 
        self.trg = Embeddings(self.trg_vocab, embed_size, device)
        self.src = Embeddings(self.src_vocab, embed_size , device)
        pe = PositionalEncoding(embed_size, drop_out)
        self.trg_embed = nn.Sequential(self.trg, copy.deepcopy(pe))
        self.src_embed = nn.Sequential(self.src, copy.deepcopy(pe))
        self.pad_idx = pad_idx
        self.encoder = Encoder(n_heads, embed_size, value_size, drop_out)
        self.decoder = Decoder(n_heads, embed_size, value_size, drop_out) 
        self.generator = PredictWord(embed_size, TV) 

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.trg_embed(tgt), memory, src_mask, tgt_mask)

class Encoder(nn.Module):
    def __init__(self, n_heads, embed_size, value_size, drop_out):
        super(Encoder, self).__init__()
        self.m_attn = MultiHeadAttention(n_heads, embed_size, value_size)
        self.ff = FeedForward(embed_size, drop_out)
        self.attn_resid = ResidualConnection( drop_out)
        self.ff_resid = ResidualConnection( drop_out)
        self.ln = LayerNorm()

    def forward(self, x, mask):
        attention_layer = self.attn_resid(x, lambda x:  self.m_attn(x,x,x, mask)) 
        ff_layer = self.ff_resid(attention_layer, self.ff) 
        x = self.ln(ff_layer)
        return ff_layer

class Decoder(nn.Module):
    def __init__(self, n_heads, embed_size, value_size,  drop_out):
        super(Decoder, self).__init__()
        
        self.src_attn = MultiHeadAttention(n_heads, embed_size, value_size)
        self.self_attn = MultiHeadAttention(n_heads, embed_size, value_size)
        self.ff = FeedForward(embed_size, drop_out)
        self.attn_self_resid = ResidualConnection( drop_out)
        self.attn_src_resid = ResidualConnection(drop_out)
        self.ff_resid = ResidualConnection( drop_out)
        self.ln = LayerNorm()


    def forward(self, x, mem_position, src_mask, trg_mask):
        x = x.float()
        m = mem_position
        x = self.attn_self_resid(x, lambda x: self.self_attn(x, x,x)) 
        
        
        
        x = self.attn_src_resid(x, lambda x: self.src_attn(x, m, m))
        ff_layer = self.ff_resid(x, self.ff) 
        ff_layer = self.ln(ff_layer) 
        return ff_layer


def data_gen(V, batch, nbatches):
        "Generate random data for a src-tgt copy task."
        for i in range(nbatches):
            data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
            data[:, 0] = 1
            src = torch.autograd.Variable(data, requires_grad=False)
            tgt = torch.autograd.Variable(data, requires_grad=False)
            yield Batch(src, tgt, 0)


if __name__ == "__main__":

    V = 11
    data = data_gen(V, BATCH_SIZE, 100)
    d = next(data)

    pad = SRC.stoi['<pad>']
    pad = TRG.stoi['<pad>']
    VS  = len(SRC)
    VT  = len(TRG)
    src = TEST_BATCH.src[0].permute((1, 0))
    trg = TEST_BATCH.trg[0].permute((1,0))
    b = mc.Batch(src, trg)

    
    #these are the batch sizes
    #print("#these are the batch sizes")
    #print(b.src.size())
    #print(b.trg.size())
    #print(b.trg_y.size())


    #model_r = Transformer(VS, VT, EMBED_SIZE, VALUE_SIZE, NHEADS, pad_idx = pad,  drop_out = DROPOUT)
    model_t = Transformer(V, V, EMBED_SIZE, VALUE_SIZE, NHEADS, pad_idx = 0,  drop_out = DROPOUT)

    for p in model_t.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

#    for p in model_r.parameters():
#        if p.dim() > 1:
#            nn.init.xavier_uniform_(p)
#    
    result = model_t(d.src, d.trg, d.src_mask, d.trg_mask)


    #result = model_r(b.src, b.trg, b.src_mask, b.trg_mask)
    #print(result.size())
     
    

import copy
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from math import inf
from utils import attention,  copies

'''
1.feed forward layer class DONE 
2.residual cnnections class DONE 
3.layer norm class DONE 
4.multihead attention
5.single head attention 
6.decoder 
7.encoder 
8.model embeddings
9.positional encoding class
10. imlement teacher forcing for training
get rid of teacher training for testing
11. mask class 

the encoder consists of two parts. 
x = (positional + multihead attention)
 y = (x + feed forward(x))`

the decoder consitst of three parts
x = (positional + masked multihead attention)
 y = (x + multihead attention(x, key value)`
x = (y + feedforward(y)
'''

###CLASSES ### HNLP = classes used from the transformer in pytorch guide from harvard nlp 

#HNLP
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=1):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & torch.autograd.Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
#HNLP
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).double()
        positions_2  = torch.arange(0 , d_model, 2).numpy() 
        denom = -(np.log(10000.0) / d_model)
        norm_p = denom * positions_2 
        norm_p = torch.from_numpy(norm_p)
        div_term = torch.exp(norm_p )
        lost = position * div_term
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + torch.autograd.Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, vocab, embed_size, device, batch_first = True):
        super(Embeddings, self).__init__()
        self.bf = batch_first
        self.embed = nn.Embedding(vocab, embed_size ).to(device)
        self.embed_size = embed_size
    def forward(self, x):
        #dvide by sqrt of embed size as described in paper
        sqrt = np.sqrt(self.embed_size)
        embeddings = self.embed(x) / sqrt

        if self.bf == True:
            #embeddings = embeddings.permute(1, 0, 2)
            pass
        return embeddings

class PredictWord(nn.Module):
    def __init__(self, embed_size, vocab):
        super(PredictWord, self).__init__()
        self.proj = nn.Linear(embed_size, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class FeedForward(nn.Module):
    """
    two layers of feed forward: embed -> inter -> embed
    nonlinearity and then dropout inbetween
    """
    def __init__(self, embed_size, dropout_rate):
        super(FeedForward, self).__init__()
        self.embed_size = embed_size 
        self.inter_size = embed_size * 4
        self.l1 = nn.Linear(embed_size, self.inter_size)
        self.l2 = nn.Linear(self.inter_size, embed_size)
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(p = dropout_rate)

    def forward(self, tensor):
        x = self.l1(tensor)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.l2(x)
        return x

class LayerNorm(nn.Module):
    '''
    because batch norm sucks
    '''
    def __init__(self):
        super(LayerNorm, self).__init__()
        self.layer_size = None
    
    def forward(self, tensor):
        ubar = tensor.mean(-1, keepdim = True)
        stdev = tensor.std(-1, keepdim = True)
        self.layer_size = torch.Tensor([tensor.size(0)])
        return (tensor - ubar) / stdev 
        
class ResidualConnection(nn.Module):
    def __init__(self, dropout_rate):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(p = dropout_rate)
        self.layer_norm = LayerNorm()

    def forward(self, x, sublayer: nn.Module):
        y = self.dropout(sublayer(x))
        y = self.layer_norm(x + y)
        return y 
#HNLP --- Unfortunaley my attention function had some bugs where the shapes coming out were (B,B,S,EMBED)--- fixing in the future
class MultiHeadAttention(nn.Module):
    
    def __init__(self, h, d_model, value_size, dropout = 0.01):
        super(MultiHeadAttention, self).__init__()
      
        if d_model % h != 0:
            raise Exception("You need cant have this number of heads")

        self.d_k = d_model // h
        self.h = h
        self.linears = copies(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask = None):

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]


        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

#HNLP
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

if __name__ == "__main__":
    torch.manual_seed(1)
    #(batch, sent_len, embed_size)
    x = torch.rand(1, 4, 6)
    y = torch.rand(1,4,6)
    #c, d = attention(x,y,y)
    #mhd = MultiHeadAttention(2,6,6)
    #print(mhd(x,y,y))

    ln = LayerNorm()

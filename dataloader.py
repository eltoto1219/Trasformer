import torchtext 
import numpy as np
import torch
import torch.nn as nn

#TRAIN_EN = "./data/train.en"  
#TRAIN_VI = "./data/train.vi" 
#TEST_EN  = "./data/tst2013.en"
#TEST_VI  = "./data/tst2013.vi"
TRAIN = "/home/antonio/Desktop/projects/transformer/data/train_mini"
EXTS = (".en", ".vi")

SOS, EOS, PAD  = "<s>", "</s>", "<pad>"
MAX_LEN = 64 
MIN_FREQ = 2
BATCH_SIZE=1
EMBED_SIZE = 256

def tokenizer(sent): 
    sent = [word for word in sent.split(" ")]
    return sent

src = torchtext.data.Field(tokenize=tokenizer,
            pad_token = PAD,
            include_lengths = True) 
            
trg = torchtext.data.Field(tokenize=tokenizer, 
            init_token = SOS, 
            eos_token = EOS, 
            pad_token = PAD,
            include_lengths = True) 

train = torchtext.datasets.TranslationDataset(
        path = TRAIN, exts = ('.en', '.vi'), 
        fields = (src, trg))

src.build_vocab(train.src, min_freq = MIN_FREQ )
trg.build_vocab(train.trg, min_freq = MIN_FREQ)

train_iter = torchtext.data.BucketIterator(
    dataset= train, 
    batch_size=BATCH_SIZE,
    sort_key=lambda x: (len(x.src), len(x.trg))
)

TEST_BATCH  = next(iter(train_iter))
SRC = src.vocab
TRG = trg.vocab
TRAIN_ITER = train_iter

if __name__ == "__main__":
    print(trg.vocab.stoi["</s>"])
    print(TEST_BATCH.trg[0])
    exit()
    nwords = len(SRC)
    embeds = Embeddings(nwords, EMBED_SIZE)
    embs = embeds(TEST_BATCH.src[0])
    src = TEST_BATCH.src[0]
    #print(embs.size())
    pad = 0
    src_mask = (src != pad).unsqueeze(-2)
    
    print(trg.size())
    


######################################################
#INCASE I EVER DECIDE TO WRITE A CUSTOM DATASET CLASS#
######################################################

#def read_files(path, max_len = MAX_LEN):
#    data = []
#    with open(path, "r") as file:
#        for line in file: 
#            line = line.strip("\n")
#            if max_len < len(line.split(" ")):
#                pass
#            data.append(line)
#    return data
#train_en = read_files(TRAIN_EN) 
#train_vi = read_files(TRAIN_VI)
#test_en =  read_files(TEST_EN)
#test_vi =  read_files(TEST_VI)
#class vocab(torchtext.datasets.TranslationDataset):

#class vocab:
#    def __init__(self, path, exts,  max_len, min_freq, sos, eos, pad, batch_size):
#        self.path = path
#        self.max_len = max_len
#        self.min_freq = min_freq
#        self.sos, self.eos, self.pad = sos, eos, pad 
#        self.batch_size
#        self.exts = exts
#
#    def tokenizer(sent): 
#        sent = [word for word in sent.split(" ")]
#        return sent
#
#    def batch_size(size):
#        self.batch_size = size

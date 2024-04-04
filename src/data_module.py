import sys; sys.path.append('.')

import itertools
import torch
from torch.nn import functional as F
from src.parameters import Parameters
from src.tokenizer import Tokenizer
from src.byte_pair_tokenizer import BytePairTokenizer

class DataModule(Parameters):
    def __init__(self, root="./data", num_workers=4, **kwargs):
        self.save_parameters(**kwargs)
    
    def get_dataloader(self, train): raise NotImplemented

    def train_dataloader(self): return self.get_dataloader(train=True)
    def val_dataloader(self): return self.get_dataloader(train=False)


class AnimeQuoteData(torch.utils.data.Dataset):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
    
    def __len__(self): return sum(1 for _ in open(self.filename, 'r'))

    def __getitem__(self, idx):
        with open(self.filename, 'r') as f:
            line = next(itertools.islice(f, idx, idx+1))
            line = line.strip()
        return line
        
    
class AnimeQuoteModule(DataModule):
    def __init__(self, tokenizer,
                 train_filename, val_filename=None,
                 num_steps=9, batch_size=4, num_workers=8):
        super().__init__()
        self.save_parameters(train_filename=train_filename,
                         val_filename=val_filename,
                         tokenizer=tokenizer,
                         batch_size=batch_size,
                         num_workers=num_workers,
                         num_steps=num_steps)
    
        tokenizer.max_sequence_length = num_steps
    
    def _tokenize(self, batch):
        x = [line for line in batch]
        x = self.tokenizer(x)
        x = torch.tensor(x)
        return x[:, :-1], *self.create_mask(x[:, :-1]), x[:, 1:]
    
    def get_dataloader(self, train):
        filename = self.train_filename if train else self.val_filename
        if filename is None: return
        dataset = AnimeQuoteData(filename=filename)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size,
                                           num_workers=self.num_workers, collate_fn=self._tokenize)
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones((sz,sz)))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask==0, -float("inf")).masked_fill(mask==1, float(0.))
        return mask

    @staticmethod
    def create_mask(tgt):
        tgt_seq_len = tgt.shape[1]
        tgt_mask = AnimeQuoteModule.generate_square_subsequent_mask(tgt_seq_len)
        tgt_padding_mask = tgt==0
        return tgt_mask, tgt_padding_mask

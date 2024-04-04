import torch, math
from torch import nn
from torch.nn import functional as F

import math
from src.module import Module

class CausalSelfAttention(nn.Module):
    # def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = nn.MultiheadAttention(**kwargs,
                                         batch_first=True
                                         )

    def forward(self, x, mask, padding_mask):
        output, attention_weights = self.mha(query=x, key=x, value=x,
                                             attn_mask=mask, key_padding_mask=padding_mask
                                             )
        self.attention_weights = attention_weights
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, max_len=1000):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        x = torch.arange(max_len, dtype=torch.float32)\
        .reshape(-1,1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)
    
    def forward(self, x):
       return x + self.P[:, :x.shape[1], :].to(x.device)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hiddens, ffn_depth, dropout):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(num_hiddens, ffn_depth),
            nn.ReLU(),
            nn.Linear(ffn_depth, num_hiddens),
        )

    def forward(self, x):
        return self.seq(x)

class AddNorm(nn.Module):
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.layer_norm = nn.LayerNorm(norm_shape)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, y):
        return self.layer_norm(x+self.dropout(y))


class DecoderBlock(nn.Module):
    def __init__(self, num_hiddens, ffn_depth,
                 num_heads, dropout, use_bias=False):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(embed_dim=num_hiddens, num_heads=num_heads,
                                            dropout=dropout, bias=use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFeedForward(num_hiddens=num_hiddens, ffn_depth=ffn_depth, dropout=dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, x, mask, padding_mask):
        output = self.causal_self_attention(x, mask, padding_mask)
        x = self.addnorm1(x, output)
        x = self.addnorm2(x, self.ffn(x))
        return x

class Decoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, ffn_depth,
                 num_heads, num_blocks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens, padding_idx=0)
        self.pos_encoding = PositionalEncoding(num_hiddens)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module("block%d"%i, DecoderBlock(num_hiddens, ffn_depth,
                                                             num_heads, dropout, use_bias))
        
    def forward(self, x, mask, padding_mask):
        x = self.pos_encoding(self.embedding(x) * math.sqrt(self.num_hiddens))
        x = self.dropout(x)
        # self.attention_weights = [[None]*len(self.blocks) for _ in range(2)]
        for i, block in enumerate(self.blocks):
            x = block(x, mask, padding_mask)
            # self.attention_weights[0][i] = block.causal_self_attention.attention_weights
            # self.attention_weights[1][i] = block.cross_attention.attention_weights
        return x

class Transformer(Module):
    def __init__(self, target_vocab_size,
                 num_hiddens, ffn_depth, num_heads, num_blocks,
                 dropout, use_bias=False, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        
        self.decoder = Decoder(vocab_size=target_vocab_size,
                               num_blocks=num_blocks, num_hiddens=num_hiddens,
                               num_heads=num_heads, ffn_depth=ffn_depth,
                               dropout=dropout, use_bias=use_bias)
        self.final_layer = nn.Linear(num_hiddens, target_vocab_size)
    
    def forward(self, x, mask, padding_mask):
        x = self.decoder(x, mask, padding_mask)
        logits = self.final_layer(x)
        return logits
    
    def criterion(self, pred, label, averaged=True):
        pred = pred.reshape(-1, pred.shape[-1])
        label = label.reshape(-1,)
        loss = F.cross_entropy(pred, label, reduction="mean" if averaged else "none")
        mask = (label.reshape(-1) != 0).type(torch.float32)
        return (loss*mask).sum() / mask.sum()
    
    def acc(self, pred, label, averaged=True):
        pred = pred.reshape(-1, pred.shape[-1])
        pred = pred.argmax(dim=-1).type(label.dtype)
        compare = (pred==label.reshape(-1)).type(torch.float32)
        return compare.mean() if averaged else compare

    def metrics(self): return [self.acc]
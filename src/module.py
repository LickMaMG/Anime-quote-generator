import sys; sys.path.append('.')

import torch
from torch import nn

from src.parameters import Parameters

class Module(nn.Module, Parameters):
    def __init__(self, **kwargs) -> None:
        super(Module, self).__init__()
        self.save_parameters(**kwargs)

    def criterion(self, pred, label): raise NotImplemented

    def forward(self, x): raise NotImplemented

    def training_step(self, batch):
        criterion = self.criterion
        pred = self(*batch[:-1])
        loss = criterion(pred, batch[-1])
        metrics = self.compute_metrics(pred, batch[-1])
        return loss, metrics
    
    def validation_step(self, batch):
        criterion = self.criterion
        pred = self(*batch[:-1])
        loss = criterion(pred, batch[-1])
        metrics = self.compute_metrics(pred, batch[-1])
        return loss, metrics
    
    def metrics(self): raise NotImplemented

    def compute_metrics(self, pred, label):
        res = {}
        for metric in self.metrics():
            res[metric.__name__] = metric(pred, label)
        return res
    
    def configure_optimizers(self): return torch.optim.Adam(params=self.parameters(), lr=self.lr)

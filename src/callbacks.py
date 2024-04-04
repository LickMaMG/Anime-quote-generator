import torch, cv2, torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

writer.add_graph(net, img_tensor/255.)
writer.close()
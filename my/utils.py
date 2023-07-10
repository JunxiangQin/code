import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.utils.data import Dataset
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import random_split


# SNN
def init_max_weights(module):
    for m in module.modules():
        if type(m) == nn.Linear:
            stdv = 1./ math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()

def SNN_Block(dim1, dim2, dropout=0.25):
    """
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )

class SNN(nn.Module):
    def __init__(self, omic_input_dim, model_size_omic='small', n_classes=4 ):
        super(SNN, self).__init()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 512]}

        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
            self.classifier = nn.Linear(hidden[-1], n_classes)
            init_max_weights(self)

    def forward(self, molecule, device):
        x = molecule.to(device=device, dtype=torch.float32)
        h_omic = self.fc_omic(x)
        h = self.classifier(h_omic)
        return h

#






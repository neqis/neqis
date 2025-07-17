import torch
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
        

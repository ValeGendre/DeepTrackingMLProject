import torch
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.C1 = nn.Conv2d(1, 32, 3)            
        self.C2 = nn.Conv2d(32, 32, 3)
        self.Norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=1)
        

    def forward(self, x):
        x = F.relu(self.C1(x))
        x = self.Norm(self.C2(x))
        return x

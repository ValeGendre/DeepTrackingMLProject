import os
os.environ["OMP_NUM_THREADS"] = "1"

import gzip
import pandas
import time
import numpy

from IPython import display

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18
import torchvision.transforms as T

import matplotlib
matplotlib.rcParams['figure.figsize'] = (9.0, 7.0)
from matplotlib import pyplot

from resources.dataset import MiceData #jetentendspasvscode

from models.model import DCFNet
from models.gaussian import gaussian_shaped_labels

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Parameters init
nb_epoch = 10
learning_rate = 0.01
momentum = 0.9
batch_size = 32
weight_decay = 5e-5

# Path to all_frames
root_dir = ""
# Number of consecutive frames
n = 2

# Loading the data
miceset = MiceData(root_dir=root_dir,n=n)
miceset_loader = DataLoader(miceset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)



crop_sz = 125
output_sz = 121

lambda0 = 1e-4
padding = 2.0
output_sigma_factor = 0.1

output_sigma = crop_sz / (1 + padding) * output_sigma_factor
y = gaussian_shaped_labels(output_sigma, [output_sz, output_sz])  #TODO: make compatible with new gaussian function
yf = torch.rfft(torch.Tensor(y).view(1, 1, output_sz, output_sz).to(DEVICE), signal_ndim=2)
# cos_window = torch.Tensor(np.outer(np.hanning(crop_sz), np.hanning(crop_sz))).cuda()  # train without cos window

model = DCFNet(yf,lambda0)
model.to(DEVICE)
criterion = nn.MSELoss(size_average=False).to(DEVICE)  #TODO: use Magali's
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)
target = torch.Tensor(y).to(DEVICE).unsqueeze(0).unsqueeze(0).repeat(batch_size * 1, 1, 1, 1)  # for training

model.train()

for i_epoch in range(nb_epoch):
    #for i, (template, search) in enumerate(miceset_loader):
    for template, search in miceset_loader:  #size of template [32(batch size),250,250(dimensions)]
        # measure data loading time
        #data_time.update(time.time() - end)

        template = template.unsqueeze(0)     #[1,32(batch size),250,250(dimensions)]
        template = template.permute(1,0,2,3)    #[32(batch size),1,250,250(dimensions)]
        #TODO: clamping
        #TODO: normalisation

        search = search.unsqueeze(0)
        search = search.permute(1,0,2,3)
        template = template.to(DEVICE)
        search = search.to(DEVICE)
        output = model(template.float(), search.float())
        loss = criterion(output, target)/template.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

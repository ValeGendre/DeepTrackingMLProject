import os
#os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import gzip
import pandas
import time
import numpy
import matplotlib.pyplot as plt
import tqdm

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

from resources.dataset import MiceData

from models.model import DCFNet
from models.gaussian import fft_label

from utils.misc import *

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Parameters init
nb_epoch = 1
learning_rate = 0.01
momentum = 0.9
batch_size = 1
weight_decay = 5e-5

# Path to all_frames
root_dir = "resources/data"
# Number of consecutive frames
n = 2

# Loading the data
miceset = MiceData(root_dir=root_dir,n=n)
miceset_loader = DataLoader(miceset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)



crop_sz = 250
output_sz = 51

lambda0 = 1e-4
padding = 2.0
output_sigma_factor = 0.1

output_sigma = crop_sz / (1 + padding) * output_sigma_factor

model = DCFNet(lambda0)
model.to(DEVICE)
criterion = nn.MSELoss().to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay)

model.train()


for i_epoch in range(1, nb_epoch+1):
    losses = []
    t0 = time.time()
    for template, search in tqdm.tqdm(miceset_loader):
        optimizer.zero_grad()

        # Generate random bounding box 
        bb = rd_BB()

        # Compute Gaussian label
        initial_label, initial_label_fft = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], bb, (250, 250), DEVICE)

        # Input Data
        template = template.unsqueeze(0)     #[1,32(batch size),250,250(dimensions)]
        template = template.permute(1,0,2,3)    #[32(batch size),1,250,250(dimensions)]

        search = search.unsqueeze(0)
        search = search.permute(1,0,2,3)


        template = template.to(DEVICE)
        search = search.to(DEVICE)

        # Forward Tracking 
        forward_tracking = model(template, search, initial_label_fft) # The label is the randomly initialized bounding box

        """
        to_plot1 = initial_label.detach().cpu()[0, 0]
        to_plot2 = forward_label.detach().cpu()[0, 0]
        plt.figure(tight_layout=True)
        plt.subplot(2,2,1)
        plt.imshow(template.cpu().numpy()[0, 0])
        plt.subplot(2,2,2)
        plt.imshow(to_plot1)
        plt.subplot(2,2,3)
        plt.imshow(search.cpu().numpy()[0, 0])
        plt.subplot(2,2,4)
        plt.imshow(to_plot2)
        plt.show()
        """
        
        numpy_label = forward_tracking.detach().cpu().numpy()[0, 0]
        M, N = numpy_label.shape
        idx = np.argmax(numpy_label)
        cy = min(int(idx % N), 200)
        cx = min(int((idx - cy)/N), 200)


        bb = [max(cx - 25, 0), max(cy - 25, 0), 51, 51]

        _, pseudo_label = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], bb, (250, 250), DEVICE)
        # Backward Tracking
        backward_tracking = model(search, template, pseudo_label) # The label is label generated by the forward tracking

        # Consistency loss between forward and backward tracking
        Consistency_loss = criterion(initial_label, backward_tracking)
        Consistency_loss.backward()
        optimizer.step()
        losses.append(Consistency_loss.detach().cpu().numpy())
    torch.save(model.state_dict(), f'Training3_model_epoch{i_epoch}')
    print(f'Epoch {i_epoch}/{nb_epoch} : loss = {np.mean(losses)}, it took : {time.time() - t0:.3g} s')

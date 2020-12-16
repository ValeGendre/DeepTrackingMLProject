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
n = 3

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
    for P1, P2, P3 in tqdm.tqdm(miceset_loader):
        optimizer.zero_grad()

        # Generate random bounding box 
        bb = rd_BB()

        # Compute Gaussian label
        T1, T1_fft = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], bb, (250, 250), DEVICE)

        # Input Data
        P1 = P1.unsqueeze(0)     #[1,32(batch size),250,250(dimensions)]
        P1 = P1.permute(1,0,2,3)    #[32(batch size),1,250,250(dimensions)]

        P2 = P2.unsqueeze(0)
        P2 = P2.permute(1,0,2,3)

        P3 = P3.unsqueeze(0)
        P3 = P3.permute(1,0,2,3)

        P1 = P1.to(DEVICE)
        P2 = P2.to(DEVICE)
        P3 = P3.to(DEVICE)

        # Forward Tracking 
        R2 = model(P1, P2, T1_fft) # The label is the randomly initialized bounding box
        
        numpy_label = R2.detach().cpu().numpy()[0, 0]
        M, N = numpy_label.shape
        idx = np.argmax(numpy_label)
        cy = min(int(idx % N), 200)
        cx = min(int((idx - cy)/N), 200)


        bb = [max(cx - 25, 0), max(cy - 25, 0), 51, 51]

        _, T2_fft = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], bb, (250, 250), DEVICE)

        R3 = model(P2, P3, T2_fft)

        numpy_label = R3.detach().cpu().numpy()[0, 0]
        M, N = numpy_label.shape
        idx = np.argmax(numpy_label)
        cy = min(int(idx % N), 200)
        cx = min(int((idx - cy)/N), 200)


        bb = [max(cx - 25, 0), max(cy - 25, 0), 51, 51]

        _, T3_fft = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], bb, (250, 250), DEVICE)

        # Bacward Tracking

        Rh2 = model(P3, P2, T3_fft)

        numpy_label = Rh2.detach().cpu().numpy()[0, 0]
        M, N = numpy_label.shape
        idx = np.argmax(numpy_label)
        cy = min(int(idx % N), 200)
        cx = min(int((idx - cy)/N), 200)


        bb = [max(cx - 25, 0), max(cy - 25, 0), 51, 51]

        _, Th2_fft = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], bb, (250, 250), DEVICE)

        Rh1 = model(P2, P1, Th2_fft)

        # Consistency loss between forward and backward tracking
        Consistency_loss = criterion(T1, Rh1)
        Consistency_loss.backward()
        optimizer.step()
        losses.append(Consistency_loss.detach().cpu().numpy())
    torch.save(model.state_dict(), f'Training3_model_epoch{i_epoch}')
    print(f'Epoch {i_epoch}/{nb_epoch} : loss = {np.mean(losses)}, it took : {time.time() - t0:.3g} s')

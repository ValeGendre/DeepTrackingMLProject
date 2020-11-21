
from resources.dataset import MiceData #jetentendspasvscode

import torch
from torch.utils.data import DataLoader

# Parameters init
batch_size = 32

# Path to all_frames
root_dir = ""
# Number of consecutive frames
n = 2

# Loading the data
miceset = MiceData(root_dir=root_dir,n=n)
miceset_loader = DataLoader(miceset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
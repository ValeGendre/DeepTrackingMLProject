import torch.nn as F
import numpy as np

def loss(forward_labels, backward_labels):
    losss = F.MSELoss()
    output = losss(backward_labels,forward_labels)
    #output.backward() # put in the train loop
    return output

def rd_BB():
    # Random init 
    x = np.random.choice(250)
    y = np.random.choice(250)
    dx = 30
    dy = 30
    return x, y, dx, dy


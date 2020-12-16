import torch.nn.functional as F
import numpy as np

def loss(forward_labels, backward_labels):
    output = F.mse_loss(backward_labels, forward_labels)
    #output.backward() # put in the train loop
    return output

def rd_BB():
    # Random init 
    x = np.random.choice(200)
    y = np.random.choice(200)
    dx = 51
    dy = 51
    return x, y, dx, dy

def center_bb(bb):
    x, y, dx, dy = bb
    cx = int(x + (dx + 1)/2)
    cy = int(y + (dy + 1)/2)
    return [cx, cy]


def BB_from_centers(cx, cy):
    x = min(max(cx - 25, 0), 199) # clamping values between 0 and 199
    y = min(max(cy - 25, 0), 199)
    dx = 51
    dy = 51
    return x, y, dx, dy
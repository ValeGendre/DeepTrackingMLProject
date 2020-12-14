import numpy as np 
import os
import matplotlib.pyplot as plt 
import torch


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def gaussian_shaped_labels(sigma, sz):
    x, y = np.meshgrid(np.arange(1, sz[0]+1) - np.floor(float(sz[0]) / 2), np.arange(1, sz[1]+1) - np.floor(float(sz[1]) / 2))
    d = x ** 2 + y ** 2
    g = np.exp(-0.5 / (sigma ** 2) * d)
    #g = np.roll(g, int(-np.floor(float(sz[0]) / 2.) + 1), axis=0)
    #g = np.roll(g, int(-np.floor(float(sz[1]) / 2.) + 1), axis=1)
    
    return g.astype(np.float32)

def gaussian_label(sigma, sz, bb, img_size):
    x,y,dx,dy = bb
    gaussian_shape = gaussian_shaped_labels(sigma, sz)
    label = np.zeros(img_size)
    label[y:y+dy, x:x+dx] = gaussian_shape

    label = np.roll(label, int(-np.floor(float(img_size[0]) / 2.) - 1), axis=0)
    label = np.roll(label, int(-np.floor(float(img_size[1]) / 2.) - 1), axis=1)

    return label

def fft_label(sigma, sz, cz, bb, img_size, DEVICE):
    label = torch.tensor(gaussian_label(sigma, sz, bb, img_size))
    y = label.view((1, 1, cz[0], cz[1])).to(DEVICE)
    yf = torch.rfft(y, signal_ndim=2) 
    y = torch.irfft(yf, signal_ndim=2)
    return y, yf
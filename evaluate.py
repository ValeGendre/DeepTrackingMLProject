import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.color import rgb2gray 

from models.model import DCFNet
from models.gaussian import fft_label
from utils.misc import *

weights_path = 'Training_withBB_model_epoch50'
output_path = 'Video2.txt'
lambda0 = 1e-4
crop_sz = 250
output_sz = 51

lambda0 = 1e-4
padding = 2.0
output_sigma_factor = 0.1

output_sigma = crop_sz / (1 + padding) * output_sigma_factor

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

show_img = False

model = DCFNet(lambda0)
model.load_state_dict(torch.load(weights_path))
model.to(DEVICE)
model.eval()

img_path = 'resources/data/all_frames/vid{}_{}.jpg'
Video = 2
id0 = 1
idmax = 800

c = [69, 114]
initial_bb = BB_from_centers(c[0], c[1])



label, labelf = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], initial_bb, (250, 250), DEVICE)
numpy_label = label.detach().cpu().numpy()[0, 0]
with open(output_path, 'a') as f:
    f.write('resources/data/all_frames/vid{}_{}.jpg'.format(Video, id0) + '\n')
    f.write(str(c[0]) + ' ' + str(c[1]) + '\n')


with torch.no_grad():
    for id1 in range(id0, idmax, 10):
        if show_img:
            plt.figure(tight_layout=True)
        id2 = id1 + 10

        bounding_box = BB_from_centers(c[0], c[1])
        label, labelf = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], bounding_box, (250, 250), DEVICE)

        template = rgb2gray(plt.imread(img_path.format(Video, id1))).astype(np.float32)
        search = rgb2gray(plt.imread(img_path.format(Video, id2))).astype(np.float32)

        if show_img:
            plt.subplot(2,2,1)
            plt.imshow(template, cmap='gray')
            plt.subplot(2,2,3)
            plt.imshow(search, cmap='gray')
            plt.subplot(2,2,2)
            plt.imshow(numpy_label)

        template = torch.tensor(template).view(1, 1, 250, 250).to(DEVICE)
        search = torch.tensor(search).view(1, 1, 250, 250).to(DEVICE)

        label = model.forward(template, search, labelf)
        
        
        
        numpy_label = label.detach().cpu().numpy()[0, 0]
        M, N = numpy_label.shape
        idx = np.argmax(numpy_label)
        cx = int(idx % N)
        cy = int((idx - cx)/N)

        c = [cx + 1, cy + 1]
        with open(output_path, 'a') as f:
            f.write('resources/data/all_frames/vid{}_{}.jpg'.format(Video, id2) + '\n')
            f.write(str(c[0]) + ' ' + str(c[1]) + '\n')
        
        if show_img:
            plt.subplot(2,2,4)
            plt.imshow(numpy_label)
            plt.show()

        labelf = torch.rfft(label, signal_ndim=2)



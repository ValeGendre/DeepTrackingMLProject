import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.color import rgb2gray 

from models.model import DCFNet
from models.gaussian import fft_label

weights_path = 'Training_croppedBB_model_epoch120'
output_path = 'test.txt'
crop_sz = 51
output_sz = 31

lambda0 = 1e-4
padding = 2.0
output_sigma_factor = 5

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
initial_bb = [c[0] - 25, c[1] - 25, 51, 51]



label, labelf = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], [10, 10, 31, 31], (51, 51), DEVICE)
numpy_label = label.detach().cpu().numpy()[0, 0]
with open(output_path, 'a') as f:
    f.write('resources/data/all_frames/vid{}_{}.jpg'.format(Video, id0) + '\n')
    f.write(str(c[0]) + ' ' + str(c[1]) + '\n')


with torch.no_grad():
    for id1 in range(id0, idmax, 10):
        if show_img:
            plt.figure(tight_layout=True)
        id2 = id1 + 10

        bb = [c[0] - 25, c[1] - 25, 51, 51]
        label, labelf = fft_label(output_sigma, [output_sz, output_sz], [crop_sz, crop_sz], [10, 10, 31, 31], (51, 51), DEVICE)

        template = rgb2gray(plt.imread(img_path.format(Video, id1))).astype(np.float32)
        search = rgb2gray(plt.imread(img_path.format(Video, id2))).astype(np.float32)

        template = template[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]
        search = search[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]]

        if show_img:
            plt.subplot(2,2,1)
            plt.imshow(template, cmap='gray')
            plt.subplot(2,2,3)
            plt.imshow(search, cmap='gray')
            plt.subplot(2,2,2)
            plt.imshow(numpy_label)

        template = torch.tensor(template).view(1, 1, 51, 51).to(DEVICE)
        search = torch.tensor(search).view(1, 1, 51, 51).to(DEVICE)

        label = model.forward(template, search, labelf)
        
        
        _, _, M, N = label.shape
        numpy_label = label.detach().cpu().numpy()[0, 0]
        idx = np.argmax(numpy_label)
        cx = min(max(int(idx % N) + (c[0] - 24), 25), 224)
        cy = min(max(int((idx - int(idx % N))/N) + (c[1] - 24), 25), 224)

        c = [cx, cy]
        with open(output_path, 'a') as f:
            f.write('resources/data/all_frames/vid{}_{}.jpg'.format(Video, id2) + '\n')
            f.write(str(c[0]) + ' ' + str(c[1]) + '\n')
        
        if show_img:
            plt.subplot(2,2,4)
            plt.imshow(numpy_label)
            plt.show()

        labelf = torch.rfft(label, signal_ndim=2)



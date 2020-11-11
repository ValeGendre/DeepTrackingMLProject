import torch, os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MiceData(Dataset):
    def __init__(self,root_dir):
        # Enter your own root_dir (for instance : \resources\data)
        # all_frames folder contains each frames of the data videos
        # the frames are named this way: nbvid_nbframe.jpg (f.i : vid2_652.jpg)
        # all_frames folder has to be in your root_dir path
        # Warning : do not rename all_frames folder
        self.root_dir = root_dir
        self.image_path = os.path.join(root_dir, 'all_frames') 
        # How many video do we have ? How many frames per video do we have ?
        idx = 1
        res = 0
        self.framepervidnb = []
        # Increase the index of vid{idx} and count how many pictures start with "vid{idx}"
        # If there is no such pictures, the loop ends.
        while res==0:
            vidinb = len([f for f in os.listdir(self.image_path) if f.startswith('vid{}'.format(idx)) and os.path.isfile(os.path.join(self.image_path, f))])
            if vidinb == 0:
                res=1
            else :
                self.framepervidnb.append(vidinb)
                idx +=1
        self.vidnb = idx-1

    def __len__(self):
        # For each video, the number of frames tuples we can make is the nb of frame -1
        # As we have vidnb video, the total of tuple is 'size' bellow.
        self.size = sum(self.framepervidnb) - self.vidnb
        return self.size

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError('List index out of range. There are no images for the index: the index must be between 0 and len(data)-1')
        # first_idx list contains the first indexes to beyond to a certain video
        first_idx = np.zeros((self.vidnb,))
        correction = np.zeros((self.vidnb,))
        for k in range(1,self.vidnb):
            first_idx[k] = sum(self.framepervidnb[0:k]) -1
            correction[k] = k-1
        # By making the difference between the first_idx list and the number of item asked
        tmp = first_idx - i - correction
        for k in range(1,self.vidnb+1):
            if tmp[-k] <= 0:
                vid_find = self.vidnb - k +1    
                break
        frame1_find = abs(tmp[vid_find-1]) +1 
        frame2_find = frame1_find +1
        img1_name = "vid{}_{}".format(vid_find,int(frame1_find))
        img2_name = "vid{}_{}".format(vid_find,int(frame2_find))
        img1 = plt.imread(self.image_path+"\\"+img1_name+".jpg")
        img2 = plt.imread(self.image_path+"\\"+img2_name+".jpg")
        return img1,img2

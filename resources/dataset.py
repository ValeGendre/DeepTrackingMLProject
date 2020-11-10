import torch, os,cv2
import numpy as np
from torch.utils.data import Dataset,DataLoader

class MiceData(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.image_path = os.path.join(root_dir, 'all_frames') # Warning : do not rename all_frames folder
        # How many video do we have ? How many frames per video do we have ?
        idx = 1
        res = 0
        self.framepervidnb = []
        while res==0:
            vidinb = len([f for f in os.listdir(self.image_path) if f.startswith('vid{}'.format(idx)) and os.path.isfile(os.path.join(self.image_path, f))])
            if vidinb == 0:
                res=1
            else :
                self.framepervidnb.append(vidinb)
                idx +=1
        self.vidnb = idx-1

    def __len__(self):
        self.size = sum(self.framepervidnb) - self.vidnb
        return self.size

    def __getitem__(self, i):
        pass

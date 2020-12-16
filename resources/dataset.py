import torch, os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from skimage.color import rgb2gray

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class MiceData(Dataset):
    def __init__(self,root_dir,n=2):
        # Enter your own root_dir (for instance : \resources\data)
        # all_frames folder contains each frames of the data videos
        # the frames are named this way: nbvid_nbframe.jpg (f.i : vid2_652.jpg)
        # all_frames folder has to be in your root_dir path
        # Warning : do not rename all_frames folder
        self.root_dir = root_dir
        self.image_path = os.path.join(root_dir, 'all_frames') 
        self.n = n
        if (n != 2) and (n != 3):
            raise NameError('The second argument must be n=2 or n=3 depending on how many frames are needed.')
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
        if self.n == 2 : self.size = sum(self.framepervidnb) - self.vidnb
        elif self.n == 3 : self.size = sum(self.framepervidnb) - 2*self.vidnb
        return self.size

    def __getitem__(self, i):
        n = self.n
        if i >= len(self) or i<0:
            raise IndexError('List index out of range. There are no images for the index: the index must be between 0 and len(data)-1')
        if n == 2:
            # first_idx list contains the first indexes to beyond to a certain video
            first_idx = np.zeros((self.vidnb,))
            correction = np.zeros((self.vidnb,))
            for k in range(1,self.vidnb):
                first_idx[k] = sum(self.framepervidnb[0:k]) -1
                correction[k] = k-1
            # By making the difference between the first_idx list and the number of item asked
            # Some values of tmp will become negative. Taking the index of the last negative value gives
            # the number of the video where the asked frame is.        
            tmp = first_idx - i - correction
            for k in range(1,self.vidnb+1):
                if tmp[-k] <= 0:
                    vid_find = self.vidnb - k +1    
                    break
            # the last negative value is directly linked to the number of frame
            frame1_find = abs(tmp[vid_find-1]) +1 
            frame2_find = frame1_find +1

            # formatting the name of the frames and load in img1 and img2
            img1_name = "vid{}_{}".format(vid_find,int(frame1_find))
            img2_name = "vid{}_{}".format(vid_find,int(frame2_find))
            img1 = rgb2gray(plt.imread(self.image_path+"\\"+img1_name+".jpg")).astype(np.float32)
            img2 = rgb2gray(plt.imread(self.image_path+"\\"+img2_name+".jpg")).astype(np.float32)
            return img1,img2
        elif n == 3:
            # first_idx list contains the first indexes to beyond to a certain video
            first_idx = np.zeros((self.vidnb,))
            correction = np.zeros((self.vidnb,))
            for k in range(1,self.vidnb):
                first_idx[k] = sum(self.framepervidnb[0:k]) -2
                correction[k] = 2*(k-1)
            # By making the difference between the first_idx list and the number of item asked
            # Some values of tmp will become negative. Taking the index of the last negative value gives
            # the number of the video where the asked frame is.        
            tmp = first_idx - i - correction
            for k in range(1,self.vidnb+1):
                if tmp[-k] <= 0:
                    vid_find = self.vidnb - k +1    
                    break
            # the last negative value is directly linked to the number of frame
            frame1_find = abs(tmp[vid_find-1]) +1 
            frame2_find = frame1_find +1
            frame3_find = frame2_find +1 
               
            # formatting the name of the frames and load in img1 img2 and img3
            img1_name = "vid{}_{}".format(vid_find,int(frame1_find))
            img2_name = "vid{}_{}".format(vid_find,int(frame2_find))
            img3_name = "vid{}_{}".format(vid_find,int(frame3_find))
            img1 = rgb2gray(plt.imread(self.image_path+"\\"+img1_name+".jpg")).astype(np.float32)
            img2 = rgb2gray(plt.imread(self.image_path+"\\"+img2_name+".jpg")).astype(np.float32)
            img3 = rgb2gray(plt.imread(self.image_path+"\\"+img3_name+".jpg")).astype(np.float32)
            return img1,img2,img3
        
class MiceData2(Dataset): # Pour la video 2 seulement
    def __init__(self,root_dir,n=2):
        self.root_dir = root_dir
        self.image_path = os.path.join(root_dir, 'all_frames') 
        self.n = n
        if (n != 2) and (n != 3):
            raise NameError('The second argument must be n=2 or n=3 depending on how many frames are needed.')
        # How many video do we have ? How many frames per video do we have ?
        idx = 1
        res = 0
        self.framevid2 = 891


    def __len__(self):
        if self.n == 2 : self.size = self.framevid2 - 10
        elif self.n == 3 : self.size = self.framevid2 - 20
        return self.size

    def __getitem__(self, i):
        n = self.n
        if i >= len(self) or i<0:
            raise IndexError('List index out of range. There are no images for the index: the index must be between 0 and len(data)-1')
        # Bounding box de la ieme frame
        input_path = 'vid2_labels.txt'
        line = 2*i
        f=open(input_path)
        lines=f.readlines()
        BB_line = lines[line+1]
        cx, cy = BB_line.split()
        cx = int(cx)
        cy = int(cy)
        f.close()
        
        if n == 2:
            frame1_find = i+1
            frame2_find = frame1_find +10

            # formatting the name of the frames and load in img1 and img2
            img1_name = "vid2_{}".format(int(frame1_find))
            img2_name = "vid2_{}".format(int(frame2_find))
            img1 = rgb2gray(plt.imread(self.image_path+"\\"+img1_name+".jpg")).astype(np.float32)
            img2 = rgb2gray(plt.imread(self.image_path+"\\"+img2_name+".jpg")).astype(np.float32)
            return img1,img2,cx,cy
        
        elif n == 3:
            frame1_find = i+1
            frame2_find = frame1_find +10
            frame3_find = frame2_find +10
               
            # formatting the name of the frames and load in img1 img2 and img3
            img1_name = "vid2_{}".format(int(frame1_find))
            img2_name = "vid2_{}".format(int(frame2_find))
            img3_name = "vid2_{}".format(int(frame3_find))
            img1 = rgb2gray(plt.imread(self.image_path+"\\"+img1_name+".jpg")).astype(np.float32)
            img2 = rgb2gray(plt.imread(self.image_path+"\\"+img2_name+".jpg")).astype(np.float32)
            img3 = rgb2gray(plt.imread(self.image_path+"\\"+img3_name+".jpg")).astype(np.float32)
            return img1,img2,img3,cx,cy
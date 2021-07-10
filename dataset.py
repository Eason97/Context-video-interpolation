
import numpy as np
from os import listdir
from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy
import torchvision.transforms.functional as TF


class vimeo_dataset(Dataset):
    """
       DBreader reads all triplet set of frames in a directory.
       Each triplet set contains frame 0, 1, 2.
       Each image is named frame0.png, frame1.png, frame2.png.
       Frame 0, 2 are the input and frame 1 is the output.
    """

    def __init__(self, viemo_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_train=[]
        for line in open('/home/fum16/softmax_fitst_version/vimeo_triplet/'+'tri_trainlist.txt'):
            line = line.strip('\n')
            if line!='':
                self.list_train.append(line)
        self.root=viemo_dir
        self.file_len = len(self.list_train)
    def __getitem__(self, index, is_train = True):
        # print(index)
        # print(self.list_train[index])
        if is_train:
            frame0 = Image.open(self.root+self.list_train[index] +'/'+"im1.png")
            frame1 = Image.open(self.root+self.list_train[index] +'/'+"im2.png")
            frame2 = Image.open(self.root+self.list_train[index] +'/'+"im3.png")
            i,j,h,w = transforms.RandomCrop.get_params(frame0, output_size = (256, 256))
            frame0_ = TF.crop(frame0, i, j, h, w)
            frame1_ = TF.crop(frame1, i, j, h, w)
            frame2_ = TF.crop(frame2, i, j, h, w)

            ## data augmentation

        frame0 = self.transform(frame0_)
        frame1 = self.transform(frame1_)
        frame2 = self.transform(frame2_)
        return frame0, frame1, frame2
    def __len__(self):
        return self.file_len
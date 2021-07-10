from PIL import Image
from os.path import join, isdir
from torch.utils.data import Dataset
from torchvision import transforms



class vimeo_test_dataset(Dataset):
    def __init__(self, viemo_dir):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.list_train=[]
        for line in open('/home/fum16/softmax_fitst_version/vimeo_triplet/'+'tri_testlist.txt'):
            line = line.strip('\n')
            if line!='':
                self.list_train.append(line)
        self.root=viemo_dir
        self.file_len = len(self.list_train)
    def __getitem__(self, index):
        frame0 = self.transform(Image.open(self.root+self.list_train[index] +'/'+ "im1.png"))
        frame1 = self.transform(Image.open(self.root+self.list_train[index] +'/'+"im2.png"))
        frame2 = self.transform(Image.open(self.root+self.list_train[index] +'/'+"im3.png"))
        return frame0, frame1, frame2
    def __len__(self):
        return self.file_len
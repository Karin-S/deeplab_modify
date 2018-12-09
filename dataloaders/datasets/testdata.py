from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from modeling.deeplab import *
import os
from modeling.deeplab import *
from dataloaders import custom_transforms as tr

class testset(Dataset):
    """
    test dataset
    """
    NUM_CLASSES = 21

    def __init__(self, dir):
        """
        :param dir: path to test dataset directory
        """
        super().__init__()
        self.dir = dir
        self.img_list = list(map(lambda x: os.path.join(dir, x), os.listdir(dir)))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        _img = Image.open(self.img_list[index]).convert('RGB')

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
                tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                tr.ToTensor()])
        return composed_transforms(sample)

dir = "E:\img"

if __name__ == '__main__':
    a = testset(dir)
    b = a.__len__()
    print(b)

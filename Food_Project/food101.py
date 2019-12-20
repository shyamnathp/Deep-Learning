import torch
import torchvision
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.autograd import Variable
import helper
from StratifiedSampler import StratifiedSampler
import numpy as np

FOOD_PATH = "/home/data/food-101"
IMG_PATH = FOOD_PATH+"/images"
META_PATH = FOOD_PATH+"/meta"
TRAIN_PATH = FOOD_PATH+"/train"
VALID_PATH = FOOD_PATH+"/valid"
MODEL_PATH = 'model_data/'

imagenet_stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]

# Return only images of certain classes
# not sure if labels are continuous
def get_range_indices(target, label):
    label_indices = []

    for i in range(len(target)):
        if target[i] in label:
            label_indices.append(i)

    return label_indices

class FOOD101():
    def __init__(self):
        self.train_ds, self.valid_ds, self.train_cls, self.valid_cls = [None]*4
        self.imgenet_mean = imagenet_stats[0]
        self.imgenet_std = imagenet_stats[1]
         
    def _get_tfms(self):
        train_tfms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            transforms.Normalize(self.imgenet_mean, self.imgenet_std)])
        
        valid_tfms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(self.imgenet_mean, self.imgenet_std)])        
        return train_tfms, valid_tfms            
            
    def get_dataset(self,root_dir='home/data/food-101'):
        train_tfms, valid_tfms = self._get_tfms() # transformations
        self.train_ds = datasets.ImageFolder(root=TRAIN_PATH, transform=train_tfms)
        self.valid_ds = datasets.ImageFolder(root=VALID_PATH, transform=valid_tfms)        
        self.train_classes = self.train_ds.classes
        self.valid_classes = self.valid_ds.classes

        assert self.train_classes==self.valid_classes
        return self.train_ds, self.valid_ds, self.train_classes

    
    def get_dls(self, train_ds, valid_ds, bs, n = 0, **kwargs):
        label_dataset = self.train_classes[1:n]
        train_indices = get_range_indices(self.train_ds.labels, label_dataset)
        bs = len(self.train_ds/bs)
        return (DataLoader(train_ds, batch_size=bs, sampler = StratifiedSampler(class_vector= label_dataset, batch_size=bs), shuffle=True, **kwargs),
               DataLoader(valid_ds, batch_size=bs, shuffle=False, **kwargs))  
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
from loguru import logger
import rioxarray as xr
from PIL import Image

import torchvision.transforms.functional as F

def normalize(band):
    band_min, band_max = (band.min(), band.max())
    return ((band-band_min)/((band_max - band_min)))

def image_preprocessing(image_path):

    image = xr.open_rasterio(image_path, masked=False).values
    
    # red = image[3,:,:]
    # green = image[2,:,:]
    # blue = image[1,:,:]
    # rgb_composite_n = np.dstack((red, green, blue))

    red = image[3,:,:]*255*2
    green = image[2,:,:]*255*2
    blue = image[1,:,:]*255*2
    

    rgb_image = np.stack((red, green, blue), axis=2).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_image)

    return rgb_image

class TrainDataset(Dataset):
    """
    Custom training dataset class.
    """
    def __init__(self, df_path):    
        self.df_path = df_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        image = image_preprocessing(img_path)

        # if self.data_augmentation: 
        #     image = data_augmentation(image)
        # else:
        #     image = image.values
        

        # image = np.transpose(image, (2,1,0))
        # image = torch.Tensor(image)

        image = self.transform(image)
    
        target = self.df_path.target.iloc[index]
        
        return image, target

    def __len__(self):
        return len(self.df_path)
    
class EvalDataset(Dataset):


    def __init__(self, df_path):
        
        self.df_path = df_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        image = image_preprocessing(img_path)
        image = self.transform(image)


        # image = np.transpose(image, (2,1,0))
        # image = torch.Tensor(image)        

        target = self.df_path.target.iloc[index] 

        return image, target


    def __len__(self):
        return len(self.df_path)
    
class TestDataset(Dataset):


    def __init__(self, df_path):
        
        self.df_path = df_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, index):

        img_path = self.df_path.image_path.iloc[index]
        image = image_preprocessing(img_path)
        # image = np.transpose(image, (2,1,0))
        # image = torch.Tensor(image)
        image = self.transform(image)
    
        return image

    def __len__(self):
        return len(self.df_path)   

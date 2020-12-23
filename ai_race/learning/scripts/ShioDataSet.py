from torch.utils.data import Dataset
import os
import io
import sys
import pandas as pd
#import cv2

from PIL import Image
import numpy as np
import torchvision.transforms as T
import random

LABEL_IDX = 2
IMG_IDX = 1

class MyDataset(Dataset):
   def __init__(self, csv_file_path, root_dir, width = 320, height = 240, transform=None, PILtrans=True, ORGtrans=True):
      self.image_dataframe = pd.read_csv(csv_file_path,engine='python')
      self.root_dir = root_dir
      self.width = width
      self.height = height
      self.randrange = 7
      self.transform = T.Compose([
         T.ColorJitter(0.5, 0.5, 0.5, 0.5),
         T.ToTensor(),
         T.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
         #T.RandomErasing()
      ])
      self.PILtrans = PILtrans
      self.ORGtrans = ORGtrans
      
   def __len__(self):
      return len(self.image_dataframe)
   
   def original_transform(self,image):
      dat_RGB = np.array(image)
      #random data input into each edge of image     
      rnd_to = np.random.rand(self.randrange,self.width,3)*255
      #rnd_to = np.random.rand(100,self.width,3)*255
      rnd_lo = np.random.rand(self.randrange,self.width,3)*255
      rnd_le = np.random.rand(self.height,self.randrange,3)*255
      rnd_ri = np.random.rand(self.height,self.randrange,3)*255
      dat_RGB[0:self.randrange,:,:] = rnd_to
      #dat_RGB[0:100,:,:] = rnd_to
      dat_RGB[-self.randrange:,:,:] = rnd_lo
      dat_RGB[:,0:self.randrange,:] = rnd_le
      dat_RGB[:,-self.randrange:,:] = rnd_ri

      #randomly enroll
      rnd_n1 = int(random.uniform(-self.randrange,self.randrange))
      rnd_n2 = int(random.uniform(-self.randrange,self.randrange))
      np.roll(dat_RGB, rnd_n1, axis=0)
      np.roll(dat_RGB, rnd_n2, axis=1)
      image = Image.fromarray(dat_RGB)

      return image

   def __getitem__(self, idx):
      label =[0]*3
      label=self.image_dataframe.iat[idx, LABEL_IDX]
      img_name = self.image_dataframe.iat[idx, IMG_IDX]
      
      image = Image.open(img_name).convert('RGB')
      if self.transform:
         if self.ORGtrans:
            image = self.original_transform(image)
            if self.PILtrans:
               image = self.transform(image)
            else:
               image = T.ToTensor()(image)
         elif self.PILtrans:
            image = self.transform(image)
         else:
            image = T.ToTensor()(image)
            
      
      return image, label

import os
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms as T 
from PIL import Image
import torch
import csv
from random import shuffle, sample
from numpy.random import choice
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')

import openpyxl as pxl
#import cv2
import pandas as pd
from sklearn.model_selection import KFold
#from skimage import img_as_ubyte
from torchvision import utils as vutils
import time
from scipy import ndimage

class grading_dataset(data.Dataset):
    def __init__(self, train=False, val=False, test=False, test_tta=False, all=False, fold_index=0):
        self.train = train
        self.val = val
        self.test = test
        self.test_tta = test_tta
        self.all = all
        self.data_path = 'data/C. Diabetic Retinopathy Grading/'

        if train or val or all:
            self.data_subpath = '1. Original Images/a. Training Set/'
            label_file = '2. Groundtruths/a. DRAC2022_ Diabetic Retinopathy Grading_Training Labels.csv'
        else:
            self.data_subpath = '/home/feng/hjl/DRAC/data/DRAC2022_Testing_Set/C. Diabetic Retinopathy Grading/1. Original Images/b. Testing Set/'
        self.images = []
        image_lists = [[] for _ in range(3)]

        if test or test_tta:
            for i in range(len(os.listdir(self.data_subpath))):
                filename = os.listdir(self.data_subpath)[i]
                self.images.append([self.data_subpath + filename, -1, filename])

        elif train or val:
            csv_data = pd.read_csv(self.data_path + label_file)
            self.label_dict = {}
            for index, row in csv_data.iterrows():
                image_id = row['image name']
                grade = int(row['DR grade'])
                image_lists[grade].append(image_id)

            classes = [0, 1, 2]
            for grade in classes:
                kf = KFold(n_splits=5, shuffle=True, random_state=5)
                for kk, (train_index, val_index) in enumerate(kf.split(range(len(image_lists[grade])))):
                    if kk == fold_index:
                        if self.train:
                            selected_indices = train_index
                        else:
                            selected_indices = val_index
                        print("Grade", grade, ':', len(selected_indices))
                for index in selected_indices:
                    filename = image_lists[grade][index]
                    self.images.append([self.data_path + self.data_subpath + filename, grade, filename])

        elif all:
            csv_data = pd.read_csv(self.data_path + label_file)
            self.label_dict = {}
            for index, row in csv_data.iterrows():
                image_id = row['image name']
                grade = int(row['DR grade'])
                self.images.append([self.data_path + self.data_subpath + image_id, grade, image_id])

        data_augmentation = {
            'brightness': 0.8,
            'contrast': 0.4,
            'scale': (0.8, 1.2),
            'ratio': (0.8, 1.2),
            'degrees': (-180, 180),
            'img_size': 384
        }
        if self.train:
            self.transform = T.Compose([
                T.Resize((640, 640)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(
                    brightness=data_augmentation['brightness'],
                    contrast=data_augmentation['contrast'],
                ),
                T.RandomResizedCrop(
                    size=((data_augmentation['img_size'], data_augmentation['img_size'])),
                    scale=data_augmentation['scale'],
                    ratio=data_augmentation['ratio']
                ),
                T.RandomAffine(
                    degrees=data_augmentation['degrees'],
                ),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif self.val or self.test or self.all:
            self.transform = T.Compose([
                T.Resize((data_augmentation['img_size'], data_augmentation['img_size'])),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        elif self.test_tta:
            self.transform = T.Compose([
                T.Resize((data_augmentation['img_size'], data_augmentation['img_size'])),
            ])

    def __getitem__(self, index):
        images, label, name = self.images[index]
        data = Image.open(images).convert('RGB')
        data = self.transform(data)

        return data, label, name

    def __len__(self):
        return len(self.images)


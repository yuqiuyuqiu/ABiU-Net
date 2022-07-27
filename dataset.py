"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import cv2
import os
import numpy as np
import torch
import Transforms as myTransforms
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):
    def __init__(self, im_list, label_list, transform=None):
        self.im_list = im_list
        self.label_list = label_list
        self.transform = transform

    def __getitem__(self, idx):
        img = cv2.imread(self.im_list[idx])
        img = img[:, :, ::-1]
        mask = cv2.imread(self.label_list[idx], 0)

        # Image Transformations
        if self.transform:
            [img, mask] = self.transform(img, mask)
        return (img, mask)

    def __len__(self):
        return len(self.im_list)


class LoadData(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.trainImList = list()
        self.valImList = list()
        self.trainAnnotList = list()
        self.valAnnotList = list()

    def read_file(self, file_name, train=False):
        with open(self.data_dir + '/' + file_name, 'r') as textFile:
            for line in textFile:
                line_arr = line.split()
                img_file = ((self.data_dir).strip() + line_arr[0].strip()).strip()
                label_file = ((self.data_dir).strip() + line_arr[1].strip()).strip()

                if train == True:
                    self.trainImList.append(img_file)
                    self.trainAnnotList.append(label_file)
                else:
                    self.valImList.append(img_file)
                    self.valAnnotList.append(label_file)

        return 0

    def process_data(self):
        print('Processing training data')
        return_train = self.read_file('DUTS-TR.lst', True)

        print('Processing validation data')
        return_val = self.read_file('ECSSD.lst', False)

        if return_train == 0 and return_val == 0:
            data_dict = dict()
            data_dict['trainIm'] = self.trainImList
            data_dict['trainAnnot'] = self.trainAnnotList
            data_dict['valIm'] = self.valImList
            data_dict['valAnnot'] = self.valAnnotList
            return data_dict
        return None


def setup_loaders(args):
    data_loader = LoadData(args.data_dir)
    data = data_loader.process_data()
    if data is None:
        raise ValueError('Error while pickling data. Please check.')

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)#RGB
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)#RGB

    train_transform = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.img_size, args.img_size),
        myTransforms.RandomCropResize(int(7./224.*args.img_size)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    val_transform = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.img_size, args.img_size),
        myTransforms.ToTensor()
    ])


    train_set = Dataset(data['trainIm'], data['trainAnnot'], transform=train_transform)
    val_set = Dataset(data['valIm'], data['valAnnot'], transform=val_transform)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False, drop_last=False)

    return train_loader, val_loader

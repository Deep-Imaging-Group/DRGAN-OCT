#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 13:29:32 2019

@author: tsmotlp
"""

import torch.utils.data as data
import os
from PIL import Image
from torchvision import transforms

# -------------------------------
# data is orgnized as follows:
# --data_roots
# ----imageX_dir
# ------imageX_1
# ------imageX_2
# ------ ......
# ------imageX_m
# ----imageY_dir
# ------imageY_1
# ------imageY_2
# ------ ......
# ------imageY_m
# -------------------------------


class get_data_list():
    def __init__(self, data_roots):
        self.data_roots = data_roots
        self.trainX_list, self.trainX_label_list, self.trainY_list, self.validX_list, self.validY_list, self.trainN_list = self.get_data_name_list()
        self.train_data = {'trainX':self.trainX_list, 'trainX_label': self.trainX_label_list, 'trainY': self.trainY_list, 'trainN':self.trainN_list}
        self.valid_data = {'validX':self.validX_list, 'validY': self.validY_list}
    
    def get_data_name_list(self):
        # train X image
        trainX_path = os.path.join(self.data_roots, 'trainX')
        trainX_list = [os.path.join(trainX_path, trainX) for trainX in sorted(os.listdir(trainX_path)) if self.is_image_file(trainX)]
        
        # X label image
        trainX_label_path = os.path.join(self.data_roots, 'trainX_label')
        trainX_label_list = [os.path.join(trainX_label_path, X_label) for X_label in sorted(os.listdir(trainX_label_path)) if self.is_image_file(X_label)]

        # train Y image
        trainY_path = os.path.join(self.data_roots, 'trainY')
        trainY_list = [os.path.join(trainY_path, trainY) for trainY in sorted(os.listdir(trainY_path)) if self.is_image_file(trainY)]

        # noise image
        trainN_path = os.path.join(self.data_roots, 'trainN')
        trainN_list = [os.path.join(trainN_path, noise) for noise in sorted(os.listdir(trainN_path)) if self.is_image_file(noise)]

        validX_path = os.path.join(self.data_roots, 'validX')
        validX_list = [os.path.join(validX_path, validX) for validX in os.listdir(validX_path) if self.is_image_file(validX)]

        validY_path = os.path.join(self.data_roots, 'validX_label')
        validY_list = [os.path.join(validY_path, validY) for validY in os.listdir(validY_path) if self.is_image_file(validY)]

        return trainX_list, trainX_label_list, trainY_list, validX_list, validY_list, trainN_list

    def is_image_file(self, file_name):
        return any(file_name.endswith(extension) for extension in ['.jpg', '.png', '.jpeg', '.tif', '.bmp'])


# dataset class
class dataset(data.Dataset):
    def __init__(self, data_list, args, op='train'):
        super(dataset, self).__init__()
        self.data_list = data_list
        self.op = op
        self.args = args
        
    def __getitem__(self, index):      
        if self.op == 'train':
            self.trainX = self.data_list['trainX']
            self.trainX_label = self.data_list['trainX_label']
            self.trainY = self.data_list['trainY']
            self.trainN = self.data_list['trainN']

            imageX = self.crop_img(self.load_image(self.trainX[index]), self.args.train_image_height, self.args.train_image_width)
            imageX_label = self.crop_img(self.load_image(self.trainX_label[index]), self.args.train_image_height, self.args.train_image_width)
            imageY = self.crop_img(self.load_image(self.trainY[index]), self.args.train_image_height, self.args.train_image_width)
            noise = self.crop_img(self.load_image(self.trainN[index]), self.args.train_image_height, self.args.train_image_width)

            imageX = self.transform(imageX)    # [batch, channel, 256, 256]
            imageX_label = self.transform(imageX_label)
            imageY = self.transform(imageY)    # [batch, channel, 256, 256]
            noise = self.transform(noise)      # [batch, channel, 256 / scale_factor, 256 / scale_factor]
            
            return {'imageX': imageX, 'labelX': imageX_label, 'imageY': imageY, 'imageN': noise}
            
        elif self.op == 'valid':
            self.validX = self.data_list['validX']
            self.validY = self.data_list['validY']
            
            imageX = self.load_image(self.validX[index])    # [batch, channel, 256, 256]
            imageY = self.load_image(self.validY[index])    # [batch, channel, 256, 256]

            imageX = self.crop_img(imageX, self.args.valid_image_height, self.args.valid_image_width)
            imageY = self.crop_img(imageY, self.args.valid_image_height, self.args.valid_image_width)
            
            imageX = self.transform(imageX)          # [batch, channel, 256 / scale_factor, 256 / scale_factor]
            imageY = self.transform(imageY)          # [batch, channel, 256 / scale_factor, 256 / scale_factor]
            return {'imageX': imageX, 'labelX': imageY, 'image_name': self.validX[index].split("\\")[-1]}
        
    def __len__(self):
        if self.op == 'train':
            return self.args.train_batch_size
        elif self.op == 'valid':
            return len(self.data_list['validX'])

    # basic functions
    def load_image(self, file_name):
        return Image.open(file_name).convert('L')
    
    def transform(self, image):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((.5,), (.5,))
                                   ])(image)

    def crop_img(self, image, height, width):
            return transforms.CenterCrop((height, width))(image)


def get_dataloader(args):
    data_list = get_data_list(args.train_valid_image_root)
    train_data = data_list.train_data
    valid_data = data_list.valid_data
    
    train_dataset = dataset(train_data, args, op='train')
    valid_dataset = dataset(valid_data, args, op='valid')
    
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.nThreads)
    valid_dataloader = data.DataLoader(dataset=valid_dataset, batch_size=1, shuffle=True)
    
    return train_dataloader, valid_dataloader
    
    
    
    
    
    
    
    
    
    

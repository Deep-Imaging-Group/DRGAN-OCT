#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:20:07 2019

@author: tsmotlp
"""
from torch.utils import data as Data
import os
import torch
from PIL import Image
from torchvision import transforms
from models import DRGANModel
from lib import tensor2image, save_img

class tester():
    def __init__(self, args):
        super(tester, self).__init__()
        self.args = args
        # model
        self.device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
        self.model = DRGANModel(self.args)
        self.model.setup()

    def test(self):
        # dataloader
        dataset = test_dataset(self.args)
        dataloader = Data.DataLoader(dataset)
        with torch.no_grad():
            for idx, data in enumerate(dataloader):
                self.model.set_input(data)
                self.model.forward()
                imageX, image_name = self.model.imageX.to(self.device), self.model.image_name[0]
                print(image_name)

                contentX = self.model.networks['netE_C'](imageX)
                clearX = self.model.networks['netG_C'](contentX)
                clearX = tensor2image(clearX)
                save_img(clearX, os.path.join(self.args.save_dir, self.args.test_image_root.split("/")[-1]+"_results"), image_name)


class test_dataset(Data.Dataset):
    def __init__(self, args):
        super(test_dataset, self).__init__()
        self.args = args
        self.img_dir = self.args.test_image_root
        self.imageX_list = self.get_test_list()

    def __getitem__(self, index):
        imageX = self.transform(self.load_image_file(self.imageX_list[index]))
        image_name = self.imageX_list[index].split("\\")[-1]
        return {'imageX': imageX, 'image_name': image_name}

    def __len__(self):
        return len(self.imageX_list)

    def get_test_list(self):
        imageX_list = [os.path.join(self.img_dir, imageX) for imageX in os.listdir(self.img_dir) if self.is_image_file(imageX)]
        return imageX_list

    def is_image_file(self, file):
        return any(file.endswith(extension) for extension in ['png', 'jpg', 'jpeg', 'tif', '.bmp'])

    def load_image_file(self, file):
        return Image.open(file).convert('L')

    def transform(self, image):
        return transforms.Compose([transforms.CenterCrop((self.args.test_image_height, self.args.test_image_width)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((.5,), (.5,))])(image)



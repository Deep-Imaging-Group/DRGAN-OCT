#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:20:07 2019

@author: tsmotlp
"""

import torch
from dataset import get_dataloader
from models import DRGANModel
from lib import Logger


# trainer
class trainer():
    def __init__(self, args):
        super(trainer, self).__init__()
        # parse option
        self.args = args
        # train data
        print('loading training data...')
        self.train_dataloader, self.valid_dataloader = get_dataloader(self.args)
        # model
        self.model = DRGANModel(self.args)
        self.model.setup()
        # visualizer
        self.train_logger = Logger(self.args.num_epochs, len(self.train_dataloader))
        self.valid_logger = Logger(self.args.num_epochs, len(self.valid_dataloader))

    def train_process(self, model, start_epoch):
        for epoch in range(start_epoch, self.args.num_epochs):
            for i, data in enumerate(self.train_dataloader):

                model.set_input(data)
                model.optimize_parameters(i)
                model.visualizer(self.train_logger, epoch)

            if epoch % self.args.save_epoch_freq == 0:
                model.save_networks(epoch)
            if epoch % self.args.valid_epoch_freq == 0:
                self.valid(epoch)
            model.update_learning_rate()

    # first training
    def first_train(self):
        self.train_process(self.model, self.args.start_epoch)

    # resume training
    def resume_train(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # load model parameters
        self.model.load_networks(self.model.networks, self.args.start_epoch-1)
        # train
        self.train_process(self.model, self.args.start_epoch)

    def train(self):
        if self.args.start_epoch > 1:
            print('resume training at epoch {}...'.format(self.args.start_epoch))
            self.resume_train()
        if self.args.start_epoch == 1:
            print('start first training...')
            self.first_train()

    def valid(self, epoch):
        torch.cuda.empty_cache()
        self.args.mode = 'valid'
        model = DRGANModel(self.args)
        self.args.load_epoch = epoch
        model.setup()
        with torch.no_grad():
            for i, data in enumerate(self.valid_dataloader):
                model.set_input(data)
                model.forward()
                model.visualizer(self.valid_logger, epoch)
                images = model.get_current_visual_images()
                if self.args.save_valid_images:
                    model.save_images(images)
        self.args.mode = 'train'




















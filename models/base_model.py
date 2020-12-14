#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 15:58:31 2019

@author: tsmotlp
"""
from abc import ABC, abstractmethod
from _collections import OrderedDict
import torch
import os
from lib import set_seed, tensor2image
from PIL import Image
from torchvision.utils import make_grid

class BaseModel(ABC):
    def __init__(self, args):
        self.args = args
        self.gpu_ids = self.args.gpu_ids
        self.mode = self.args.mode       # choose from [train/valid/test]
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')  # get device name: CPU or GPU
        self.networks = {}
        self.objectives = {}
        self.optimizers = {}
        self.schedulers = []
        self.metric = 0
        self.visual_images = []
        self.visual_losses = []
        self.image_name = ''
        self.nrow = 4

    def set_random_seed(self, seed):
        set_seed(seed)

    @abstractmethod
    def setup(self):
        """Set up models, loss functions, optimizers, schedulers etc.

        Parameters:
        """
        pass


    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

                Parameters:
                    input (dict): includes the data itself and its metadata information.
                """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self, idx):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def get_current_visual_images(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_images:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_visual_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        losses_ret = OrderedDict()
        for name in self.visual_losses:
            if isinstance(name, str):
                losses_ret[name] = float(getattr(self, name))  # float(...) works for both scalar tensor and float number
        return losses_ret

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for scheduler in self.schedulers:
            if self.args.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = list(self.optimizers.values())[0].param_groups[0]['lr']
        print('learning rate = %.8f' % lr)

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        save_dir = os.path.join(self.args.save_dir, 'checkpoints')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for name, net in self.networks.items():
            save_filename = '%s_epoch%d.pth' % (name, epoch)
            save_path = os.path.join(save_dir, save_filename)

            if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                torch.save(net.state_dict(), save_path)
                net.cuda(self.gpu_ids[0])
            else:
                torch.save(net.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, networks, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name, net in networks.items():
            load_filename = '%s_epoch%d.pth' % (name, epoch)
            load_path = os.path.join(self.args.save_dir, 'checkpoints', load_filename)

            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            # if you are using PyTorch newer than 0.4 (e.g., built from
            # GitHub source), you can remove str() on self.device
            state_dict = torch.load(load_path, map_location=str(self.device))
            net.cuda(self.gpu_ids[0])
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            # patch InstanceNorm checkpoints prior to 0.4
            for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
            net.load_state_dict(state_dict)


    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print('---------- Networks initialized -------------')
        for name, net in self.networks.items():
            num_params = 0
            for param in net.parameters():
                num_params += param.numel()
            if verbose:
                print(net)
            print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def visualizer(self, logger, epoch):
        images = self.get_current_visual_images()
        losses = self.get_current_visual_losses()

        image_list = [value[0] for key, value in images.items()]
        images = {'TRAINING IMAGES': make_grid(image_list, nrow=self.nrow, pad_value=255)}
        logger.log(losses=losses, images=images, epoch=epoch)

    def save_images(self, images):
        for name, image in images.items():
            save_dir = os.path.join(self.args.save_dir, 'result_images')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_filename = '%s%s_epoch%s.tif' % (name, self.image_name[0].split(".")[0], self.args.load_epoch)    # batch_size=1
            save_path = os.path.join(save_dir, save_filename)
            image = tensor2image(image)
            pil_image = Image.fromarray(image.squeeze())
            pil_image.save(save_path)




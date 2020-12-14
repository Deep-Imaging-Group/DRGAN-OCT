#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:18:23 2019

@author: tsmotlp
"""
import os
import torch
from PIL import Image
from torch.backends import cudnn
from torch.nn import init
from torch.optim import lr_scheduler
import numpy as np


def init_weights(net, init_type='normal', gain=0.2):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)



def init_net(nets, init_type='normal', gpu_ids=[]):
    for name, net in nets.items():
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.cuda(gpu_ids[0])
            # net = torch.nn.DataParallel(net, gpu_ids)
        init_weights(net, init_type)
    return nets

# 设置随机数
def set_seed(seed=2333):
    if torch.cuda.is_available():
        cudnn.benchmark = True
        torch.cuda.empty_cache()
        cudnn.deterministic = True
        print("Random Seed: ", seed)
        torch.manual_seed(seed)  # 为CPU设置随机种子
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
    else:
        print("Random Seed: ", seed)
        torch.manual_seed(seed)


def save_img(image, path, image_name):
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, image_name)
    pil_image = Image.fromarray(image.squeeze())
    pil_image.save(image_path)

def tensor2image(tensor):
    # image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
    tensor0 = (tensor[0] * 0.5 + 0.5).clamp(0, 1)
    image = 255 * tensor0.cpu().float().numpy()
    # image = 255*tensor[0].clamp(0, 1).detach().cpu().data.numpy()
    return image.astype(np.uint8)

class save_models():
    def __init__(self, args, models, epoch):
        self.model_folder = args.model_para_root
        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)
        self.models = models
        self.epoch = epoch

    def save_checkpoint(self):
        for (key, value) in self.models.items():
            checkpoint_path = self.model_folder + '{}_{}.pkl'.format(key, self.epoch)
            torch.save(value.state_dict(), checkpoint_path)
            print("Checkpoint saved to {}".format(checkpoint_path))




def dis_clip(dis, clip_value):
    # Clip weights of discriminator
    for p in dis.parameters():
        p.data.clamp_(clip_value, clip_value)


def get_scheduler(optimizer, args):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + args.epoch_count - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_iters, gamma=0.5)
    elif args.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, threshold=0.01, patience=2)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:02:24 2019

@author: tsmotlp
"""
import argparse


class configs():
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='hyper-parameters for DRGAN')
        self.parser.add_argument('--mode', required=True, help='switch to choose [train/valid/test]')
        self.parser.add_argument('--nThreads', type=int, default=4)
        self.parser.add_argument('--seed', type=int, default=2333)
        self.parser.add_argument('--gpu_ids', type=list, default=[0])

        # data related
        self.parser.add_argument('--train_valid_image_root', type=str, default='./data/train_valid')
        self.parser.add_argument('--train_image_height', type=int, default=256)
        self.parser.add_argument('--train_image_width', type=int, default=256)
        self.parser.add_argument('--valid_image_height', type=int, default=448)
        self.parser.add_argument('--valid_image_width', type=int, default=896)
        self.parser.add_argument('--test_image_root', type=str, default='./data/test')
        self.parser.add_argument('--test_image_height', type=int, default=448)
        self.parser.add_argument('--test_image_width', type=int, default=896)

        # training related
        self.parser.add_argument('--init_type', type=str, default='kaiming')
        self.parser.add_argument('--gan_type', type=str, default='lsgan')
        self.parser.add_argument('--num_epochs', type=int, default=100)
        self.parser.add_argument('--train_batch_size', type=int, default=2)
        self.parser.add_argument('--lr', type=float, default=1e-4)
        self.parser.add_argument('--num_critics', type=int, default=5)
        self.parser.add_argument('--decay_epoch', type=int, default=5)
        self.parser.add_argument('--pool_size', type=int, default=5)
        self.parser.add_argument('--lambda_noise', type=float, default=1)
        self.parser.add_argument('--lambda_cycle', type=float, default=10)
        self.parser.add_argument('--lambda_recon', type=float, default=10)
        self.parser.add_argument('--lr_policy', type=str, default='step')
        self.parser.add_argument('--lr_decay_iters', type=int, default=20)

        # resume train related
        self.parser.add_argument('--start_epoch', type=int, default=1)
        self.parser.add_argument('--save_epoch_freq', type=int, default=1)
        self.parser.add_argument('--valid_epoch_freq', type=int, default=1)
        self.parser.add_argument('--save_valid_images', type=bool, default=False)
        self.parser.add_argument('--load_epoch', type=int, default=77)
        self.parser.add_argument('--save_dir', type=str, default='outputs')

        # model related
        self.parser.add_argument('--input_dim', type=int, default=1)
        self.parser.add_argument('--output_dim', type=int, default=1)
        self.parser.add_argument('--norm', type=str, default='in')
        self.parser.add_argument('--activ', type=str, default='relu')
        self.parser.add_argument('--pad_type', type=str, default='reflect')
        self.parser.add_argument('--dim', type=int, default=64)
        self.parser.add_argument('--style_dim', type=int, default=8)
        self.parser.add_argument('--mlp_dim', type=int, default=256)
        self.parser.add_argument('--n_downsample', type=int, default=2)
        self.parser.add_argument('--n_upsample', type=int, default=2)
        self.parser.add_argument('--n_res', type=int, default=4)
        self.parser.add_argument('--skip_connect', type=bool, default=True)
        self.parser.add_argument('--n_layer_D', type=int, default=4)
        self.parser.add_argument('--num_feat_layers', type=int, default=4)

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 18:28:30 2019

@author: tsmotlp
"""
from lib import set_seed
from configs import configs
from run import trainer, tester

if __name__ == '__main__':
    set_seed(seed=123)
    configs = configs().parse()
    if configs.mode == 'train':
        trainer = trainer(configs)
        trainer.train()
    if configs.mode == 'test':
        tester = tester(configs)
        tester.test()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 16:02:24 2019

@author: tsmotlp
"""

import torch
import numpy as np
from skimage import measure as sm


def ssim(output, target):
    with torch.no_grad():
        output_np = output.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()
        bc_sz = output_np.shape[0]

        total_sum = 0.0
        for idx in range(bc_sz):
            tmp_o, tmp_t = np.transpose(output_np[idx], (1, 2, 0)), np.transpose(target_np[idx], (1, 2, 0))
            total_sum += sm.compare_ssim(tmp_o, tmp_t)
    return total_sum / bc_sz


def psnr(output, target):
    with torch.no_grad():
        output_np = output.cpu().detach().numpy()
        target_np = target.cpu().detach().numpy()
        bc_sz = output_np.shape[0]
        total_sum = 0.0
        for idx in range(bc_sz):
            tmp_o, tmp_t = np.transpose(output_np[idx], (1, 2, 0)), np.transpose(target_np[idx], (1, 2, 0))
            total_sum += sm.compare_psnr(tmp_o, tmp_t)
    return total_sum / bc_sz
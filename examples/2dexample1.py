"""
Example 2. Optimizing vertices.
"""
from __future__ import division
import os
import argparse
import glob

import torch
import torch.nn as nn
import numpy as np
from skimage.io import imread, imsave
import tqdm
import imageio
import time
import cv2

image1 = 'data/eye.png'
image2 = 'data/eye_flip.png'

image_ref_1 = torch.from_numpy(imread(image1).astype('float32') / 255.).permute(2, 0, 1)[:3, ::]
image_ref_2 = torch.from_numpy(imread(image2).astype('float32') / 255.).permute(2, 0, 1)[:3, ::]

source = nn.Parameter(image_ref_1)
flow = nn.Parameter(torch.zeros_like(image_ref_1))
target = image_ref_2

optimizer = torch.optim.Adam([source], lr=0.0001)

loop = tqdm.tqdm(range(15000))


s = nn.Sigmoid()

for i in loop:
    loop.set_description('Optimizing')
    # print(source.shape)

    flow_a = s(flow)

    source_d = source + flow_a

    loss = torch.sum(torch.abs(source_d - target))
    loss.backward()
    optimizer.step()

    out = (source_d.permute(1, 2, 0).detach().numpy() * 255.).astype(np.uint8)
    if i % 10 == 0:
        cv2.imshow("out",out)
        cv2.waitKey(1)

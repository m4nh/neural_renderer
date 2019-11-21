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


class Morph(nn.Module):
    def __init__(self, source, target):
        super(Morph, self).__init__()

        # self.source = nn.Parameter(source)
        self.register_buffer("source", source)
        self.register_buffer("target", target)
        self.flow = nn.Parameter(self.source)

        self.s1 = nn.Sigmoid()
        self.s2 = nn.Sigmoid()

    def getImage(self):
        return (np.clip((self.flow).permute(1, 2, 0).detach().numpy(), 0., 1.) * 255.).astype(
            np.uint8)

    def forward(self):
        loss =  torch.sum(torch.abs(self.flow - self.target)) + 0.1*torch.sum(torch.abs(self.flow - self.source))
        return loss


image1 = 'data/eye_side.png'
image2 = 'data/eye_flip.png'

image_ref_1 = torch.from_numpy(imread(image1).astype('float32') / 255.).permute(2, 0, 1)[:3, ::]
image_ref_2 = torch.from_numpy(imread(image2).astype('float32') / 255.).permute(2, 0, 1)[:3, ::]

source = nn.Parameter(image_ref_1, requires_grad=True)
target = image_ref_2

model = Morph(source, target)

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

loop = tqdm.tqdm(range(15000))

for i in loop:
    loop.set_description('Optimizing')
    # print(source.shape)

    loss = model()
    loss.backward(retain_graph=True)
    optimizer.step()
    print(loss)

    out = model.getImage()
    if i % 10 == 0:
        cv2.imshow("out", out)
        cv2.waitKey(1)

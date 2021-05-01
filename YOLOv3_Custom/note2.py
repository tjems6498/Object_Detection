import numpy as np
import torch
from dataset import YOLODataset
import config
import pdb
from util import generalized_intersection_over_union
import torch.nn as nn
import math
import torch
import numpy as np
import os

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, _ in dataloader:
        # pdb.set_trace()
        for i in range(3):
            mean[i] += (inputs[:,:,:,i]/255.0).mean()
            std[i] += (inputs[:,:,:,i]/255.0).std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

anchors = config.ANCHORS

TRAIN_DIR = 'E:\\Computer Vision\\data\\project\\fruit_yolov3_final\\train'

dataset = YOLODataset(
        root=TRAIN_DIR,
        anchors=anchors,
        transform=None,
        mosaic=True
    )

mean, std = get_mean_and_std(dataset)
print(mean)
print(std)
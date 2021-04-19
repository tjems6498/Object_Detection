import os
import cv2
import numpy as np
from PIL import Image
from model import YOLOv3
import torch.optim as optim
import config
import torch
from backbone.darknet53 import darknet53_model
#
# model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
# model2 = darknet53_model(1000).to(config.DEVICE)
#
# state_dict = model.state_dict()
# param_names = list(state_dict.keys())
#
# check_point = torch.load('darknet53_pretrained.pth.tar', map_location=config.DEVICE)
# pretrained_state_dict = check_point['state_dict']
# pretrained_param_names = list(check_point['state_dict'].keys())
#
# print(param_names[308:])
# # print(pretrained_param_names[:312])
# print(state_dict['layers.10.layers.3.1.bn.weight'])
# for i, param in enumerate(param_names[:312]):
#     state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
# model.load_state_dict(state_dict)
# print(state_dict['layers.10.layers.3.1.bn.weight'])
#
# # # model2.load_state_dict(check_point['state_dict'], strict=False)
# # print(model2.state_dict()['conv1.layers.0.weight'])
#
#
exist_epoch = 50
for epoch in range(exist_epoch, config.NUM_EPOCHS+exist_epoch) if exist_epoch else range(config.NUM_EPOCHS):
    print(epoch)
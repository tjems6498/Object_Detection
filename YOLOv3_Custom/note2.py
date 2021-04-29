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

# gt_bbox = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float32)
# pr_bbox = torch.tensor([[0.75, 0.75, 0.5, 0.5]], dtype=torch.float32)
#
# mse = nn.MSELoss()
# # loss = generalized_intersection_over_union(pr_bbox, gt_bbox, box_format='midpoint')
# # print(loss)
#
# print(mse(pr_bbox, gt_bbox))
# img_path = 'E:\\Computer Vision\\data\\project\\fruit_yolov3_remove\\valid\\images'
# label_path = 'E:\\Computer Vision\\data\\project\\fruit_yolov3_remove\\valid\\labels'
#
# img = os.listdir(img_path)
# label = os.listdir(label_path)
#
# for i in range(len(label)):
#     if img[i].split('.')[0] != label[i].split('.')[0]:
#         print(img[i], label[i])
#         break
#

checkpoint = torch.load('yolov3.pt')
print(checkpoint['state_dict'])
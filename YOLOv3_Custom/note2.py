import numpy as np
import torch
from dataset import YOLODataset
import config
import pdb
from util import generalized_intersection_over_union
import torch.nn as nn
import math

# gt_bbox = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float32)
# pr_bbox = torch.tensor([[0.75, 0.75, 0.5, 0.5]], dtype=torch.float32)
#
# mse = nn.MSELoss()
# # loss = generalized_intersection_over_union(pr_bbox, gt_bbox, box_format='midpoint')
# # print(loss)
#
# print(mse(pr_bbox, gt_bbox))


def solution(brown, yellow):
    x = int(((4+brown) + math.sqrt((4+brown)**2 - 16*(brown+yellow))) / 4)
    y = int((brown + yellow) / x)

    if x > y:
        return [x, y]
    else:
        return [y, x]


ans = solution(24, 24)
print(ans)



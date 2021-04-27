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

# gt_bbox = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float32)
# pr_bbox = torch.tensor([[0.75, 0.75, 0.5, 0.5]], dtype=torch.float32)
#
# mse = nn.MSELoss()
# # loss = generalized_intersection_over_union(pr_bbox, gt_bbox, box_format='midpoint')
# # print(loss)
#
# print(mse(pr_bbox, gt_bbox))

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]

new_ANCHORS = [
    [(130.83, 128.05), (180.67, 187.45), (309.66, 288.85)],
    [(74.729, 73.366), (96.186, 92.182), (87.539, 138.52)],
    [(24.245, 22.306), (39.584, 38.496), (56.897, 56.142)],
]


print(np.array(ANCHORS)*416)
print(np.array(new_ANCHORS) / 416)

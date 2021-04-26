import numpy as np
import torch
from dataset import YOLODataset
import config
import pdb
from util import generalized_intersection_over_union
import torch.nn as nn


gt_bbox = torch.tensor([[0.25, 0.25, 0.5, 0.5]], dtype=torch.float32)
pr_bbox = torch.tensor([[0.75, 0.75, 0.5, 0.5]], dtype=torch.float32)

mse = nn.MSELoss()
# loss = generalized_intersection_over_union(pr_bbox, gt_bbox, box_format='midpoint')
# print(loss)

print(mse(pr_bbox, gt_bbox))
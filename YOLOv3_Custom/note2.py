import numpy as np
import torch
from dataset import YOLODataset
import config
import pdb


a = torch.rand((2,3), requires_grad=True)

print(a)
print(a.requires_grad)

b = a.detach().clone()
print(b)
print(b.requires_grad)


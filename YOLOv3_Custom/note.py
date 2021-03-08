import torch
import numpy as np
import config
from collections import Counter
# a = torch.zeros((1,5,5,6))
#
# print(a[...,4].shape)
# print(a[...,4:5].shape)

# print(torch.arange(13).repeat(2,3,13,1).unsqueeze(-1).shape)

# print(torch.tensor([-5,0,4,3]).clamp(0))
# print(np.linspace(0,1,4))
# print(torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2))
#
# print(torch.tensor(config.ANCHORS) * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1,3,2))
# print([[] for _ in range(8)])

ground_truths = [0,0,0,0,0,1,1,2,2,3,3,4,4,4,4,4,4,4]
a = Counter([gt for gt in ground_truths])
print(a)

for key, val in a.items():
    print(key, val)
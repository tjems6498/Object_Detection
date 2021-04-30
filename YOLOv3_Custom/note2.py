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

path = 'C:\\Users\\hong\\Desktop\\obj_train_data'
txt_list = os.listdir(path)

for i in range(len(txt_list)):

    with open(os.path.join(path, txt_list[i]), 'w') as f:
        a = f.readline()
        print(a)
        break

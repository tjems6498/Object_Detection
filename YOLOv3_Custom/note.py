import os
import cv2
import numpy as np
from PIL import Image
from model import YOLOv3
import torch.optim as optim
import torch
import random

a = torch.rand((2,3,3))

print(a)
index = torch.tensor([1,0])

print(a[index])
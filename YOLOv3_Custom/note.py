import os
import cv2
import numpy as np
from PIL import Image
from model import YOLOv3
import torch.optim as optim
import torch
import random

for i in range(10):
    print(random.randint(0, 10))
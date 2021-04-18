import os
import cv2
import numpy as np
from PIL import Image
from model import YOLOv3
import torch.optim as optim
import config


model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=0.0001, weight_decay=config.WEIGHT_DECAY
)

for param_group in optimizer.param_groups:
    print(param_group)
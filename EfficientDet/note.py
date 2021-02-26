import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import os
import re


scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
print(np.tile(scales, (5, 5)).shape)
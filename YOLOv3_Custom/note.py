import os
import cv2
import numpy as np
from PIL import Image

# img = cv2.imread('E:\\Computer Vision\\data\\project\\fruit_yolov3\\train\\images\\2323.jpg')
# print(img.shape)

img = np.array(Image.open('E:\\Computer Vision\\data\\project\\fruit_yolov3\\train\\images\\2323.jpg').convert('RGB'))
print(img.shape)
import numpy as np

import os

path = 'E:\\Computer Vision\\data\\project\\fruit_yolov3\\valid\\labels'
id = os.listdir(path)

for i in id:
    with open(os.path.join(path, i), 'r') as f:
        val = f.readline()
        # print(val)
        if val == '':
            print(i)
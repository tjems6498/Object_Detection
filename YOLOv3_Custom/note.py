import torch
import numpy as np
import config
from collections import Counter

'''
노트

1. 현재 사용한 defualt Anchor box가 COCO dataset에 대한 k-means clustering에 의해 구해진 값인데 (여기서는 416으로 나눠서 Normalize된 값 사용)
사람데이터는 vertical한 박스가 많기 때문에 여기에 맞게 다시 anchor box를 조정하는 것을 고려해볼 필요가 있을 것 같다.

2. albumentation이 정말 좋은 라이브러리 이지만 box augmentation에서 회전은 안되는것 같다. (내가 모르는 건가..)





'''






















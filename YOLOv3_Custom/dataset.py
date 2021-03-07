import config
import numpy as np
import os
import pandas as pd
import torch
import pdb

from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader

from YOLOv3_Custom.utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)


ROOT = 'E:\\Computer Vision\\data\\custom'

def readId():
    id = []
    with open(ROOT+'\\train.txt', 'r') as f:
        line = f.readline()
        while line != "":
            id.append(line.split('.')[0][-6:])
            line = f.readline()
    return id


class YOLODataset(Dataset):
    def __init__(self, root, anchors, image_size=416, S=[13, 26, 52], C=20, transform=None):
        self.annotations = readId()
        self.root = root
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # tensor(9, 2)
        self.num_anchors = self.anchors.shape[0]  # 9
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        id = self.annotations[idx]
        print(id)
        # np.loadtxt : 공백 기준으로 나누고 최소 2차원 array로 반환
        # np.roll : 첫번째 열을 4번째로 이동
        image = np.array(Image.open(os.path.join(ROOT,"image",id+".jpg")).convert('RGB'))
        bboxes = np.roll(np.loadtxt(fname=os.path.join(ROOT,"annotation",id+".txt"), delimiter=" ", ndmin=2), 4, axis=1)
        bboxes[:,:4] = bboxes[:,:4] - 1e-5
        bboxes = bboxes.tolist()
        # 1e-5를 빼준 이유는 albumentation에서 box transform을 할 때 박스값 중 1이 들어가면 반환될때 1이 넘어가는 이상한 오류가 있어서 이렇게 변경함.

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # dim [(3,13,13,6),(3,26,26,6)(3,52,52,6)]  6 : (p_o, x, y, w, h, class)
        targets = [torch.zeros((self.num_anchors//3, S, S, 6)) for S in self.S]

        for box in bboxes:  # 각 스케일 셀 별 하나의 anchor box scale에 target값을 설정해주는 로직

            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)  # 한개의 박스와 9개의 anchor간의 w,h iou tensor(9,)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)  # 높은순으로 인덱스 값
            x, y, width, height, class_label = box
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale  # anchor_idx가 8이면 scale_idx가 2가되고 52x52를 의미  (0, 1, 2)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # 3개중 사용할 스케일  (0, 1, 2)
                S = self.S[scale_idx]  # anchor_idx가 8이면 52

                # 만약 x,y가 0.5라면 물체가 이미지 중앙에 있다는 의미, S가 13이면 int(6.5) -> 6이 되고 이 6x6셀이 13x13에서의 위치가 됨
                # 애초부터 txt파일에서 bbox가 0~1로 Normalize 되어있기 때문에 가능
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]  ########################### 확 인 ############################# j, i가 바뀌어야 하는게 아닌가?

                if not anchor_taken and not has_anchor[scale_idx]:  # 둘다 False(혹은 0)이어야 추가. 즉 해당 scale의 그 cell 자리에 이미 물체가 있다면 pass.
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1  # probability object = 1
                    # pdb.set_trace()
                    x_cell, y_cell = S * x - j, S * y - i  # 중심점이 있는 셀에서의 위치 0~1  (위에서 i,j구할때 int를 씌우면서 사라진 소수점 값이라고 생각하면 됨)
                    width_cell, height_cell = (width * S, height * S)  # 해당 스케일(13x13 or 26x26 or 52x52)에서의 크기를 나타냄 (당연히 1보다 클 수있음)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)  # class_label이 float으로 되어있음
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction  굳이 쓰는 이유는 뭘까

        return image, tuple(targets)



def test():
    anchors = config.ANCHORS
    transform = config.train_transforms

    dataset = YOLODataset(
        root=ROOT,
        anchors=anchors,
        transform=transform
    )
    S = [13, 26, 52]
    scaled_anchors = torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1,3,2)  # (3, 3, 2)
    '''
    scaled_anchors
    tensor([[[ 3.6400,  2.8600],
         [ 4.9400,  6.2400],
         [11.7000, 10.1400]],

        [[ 1.8200,  3.9000],
         [ 3.9000,  2.8600],
         [ 3.6400,  7.5400]],

        [[ 1.0400,  1.5600],
         [ 2.0800,  3.6400],
         [ 4.1600,  3.1200]]])
    '''
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    for x, y in loader:

        boxes = []

        for i in range(y[0].shape[1]):  # y[0].shape : (batch, 3, 13, 13, 6)
            anchor = scaled_anchors[i]
            print(anchor.shape)
            print(y[i].shape)
            boxes += cells_to_bboxes(
                y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
            )[0]  # batch 제외 (num_anchors * S * S, 6)

        boxes = nms(boxes, iou_threshold=1, threshold=0.7, box_format='midpoint')
        print(boxes)
        plot_image(x[0].permute(1,2,0).to('cpu'), boxes)



if __name__ == "__main__":
    test()





import config
import numpy as np
import os
import torch
import pdb
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from util import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)


def readId(root):
    root = os.path.join(root, 'images')
    img_ids = os.listdir(root)
    ids = [i.split('.')[0] for i in img_ids]
    return ids


class YOLODataset(Dataset):
    def __init__(self, root, anchors, image_size=416, S=[13, 26, 52], C=4, transform=None):
        self.ids = readId(root)
        self.root = root
        self.image_size = image_size
        self.transform = transform
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  # == torch.tensor(anchors).view(9,2)
        self.num_anchors = self.anchors.shape[0]  # 9
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]

        # image = np.array(Image.open(os.path.join(self.root, "images", id+".jpg")).convert('RGB'))
        try:
            image = cv2.cvtColor(cv2.imread(os.path.join(self.root, "images", id+".jpg")), cv2.COLOR_BGR2RGB)
        except:
            print(os.path.join(self.root, "images", id+".jpg"))

        # 공백 기준으로 나눔 + 최소 2차원 array로 반환
        # np.roll : 첫번째 원소를 4칸 밀고 나머지를 앞으로 끌어옴  (0 ,1 ,2 ,3 ,4) -> (1, 2, 3, 4, 0)
        # 즉 label값을 0번째에서 4번째로 이동
        bboxes = np.roll(np.loadtxt(fname=os.path.join(self.root, "labels", id+".txt"), delimiter=" ", ndmin=2), 4, axis=1)
        bboxes[:,:4] = bboxes[:,:4] - 1e-5
        # 1e-5를 빼준 이유는 albumentation에서 box transform을 할 때 bbox 값에 1이 들어가면 반환될때 1이 넘어가는 오류가 있어서 이렇게 변경함.
        bboxes = bboxes.tolist()  # 2차원 리스트

        if self.transform:
            augmentations = self.transform(image=image, bboxes=bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        # dim [(3,13,13,6),(3,26,26,6)(3,52,52,6)]  6 : (object_prob, x, y, w, h, class)
        targets = [torch.zeros((self.num_anchors//3, S, S, 6)) for S in self.S]

        for box in bboxes:  # 각 스케일 셀 별 하나의 anchor box scale에 target값을 설정해주는 로직

            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)  # 한개의 박스와 9개의 anchor간의 w,h iou tensor(9,)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)  # 내림차순 정렬의 인덱스 값
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # [False, False, False]
            for anchor_idx in anchor_indices:  # true bbox와 iou가 큰 앵커 부터
                scale_idx = anchor_idx // self.num_anchors_per_scale  # anchor_idx가 8이면 scale_idx가 2가되고 52x52를 의미 (0, 1, 2)
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale  # 각 그리드스케일 에서 사용할 3개의 anchor 스케일 (0, 1, 2)
                S = self.S[scale_idx]  # anchor_idx가 8이면 52

                # 만약 x,y가 0.5라면 물체가 이미지 중앙에 있다는 의미,
                # S가 13x13이면 int(6.5) -> 6이 되고 13x13에서 이 6x6번째 셀에 물체의 중심이 있다는 의미
                # 애초부터 txt파일에서 bbox가 0~1로 Normalize 되어있기 때문에 가능
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]  # [anchor_idx, 행(y), 열(x), object probability]

                if not anchor_taken and not has_anchor[scale_idx]:  # 둘다 False(혹은 0)이어야 추가. 즉 한박스당 3개의 스케일에 한번씩만 아래 작업수행
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1  # object probability= 1
                    # pdb.set_trace()
                    x_cell, y_cell = S * x - j, S * y - i  # 중심점이 있는 셀에서의 위치 0~1  (위에서 i,j구할때 int를 씌우면서 사라진 소수점 값이라고 생각하면 됨)
                    width_cell, height_cell = (width * S, height * S)  # 해당 스케일(13x13 or 26x26 or 52x52)에서의 비율로 나타냄 (당연히 1보다 클 수있음)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)  # class_label이 float으로 되어있음
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1
                    '''
                    68번줄부터에 대한 설명: 현재 하고있는게 한 이미지에 있는 여러 정답박스중(물론 물체가 하나라서 정답박스가 하나일 수 있음) 
                    한개의 박스에 대한 정답을 세개의 스케일(13x13  26x26  52x52)에서 해당 위치에 o_p = 1, bbox값,라벨을 설정하고 있다.
                    위 과정을 9개의 anchor box와 정답박스의 크기를 비교해서 정답박스의 크기와 가장 일치하는 순서대로 진행을 하게되는데
                    첫번째 값의 인덱스가 3라면 3번째 앵커박스 즉 32x32 의 0번째 위치에 o_p가 1이 되는것이다. -> target[1][0, 행, 열, 0] = 1
                    (위 방법처럼 bbox값 + 라벨값도 넣어줌)
                    
                    elif 구문은 해당 스케일에 대표 앵커가 정해졌지만 박스의 크기와 현재 인덱스의 anchor box의 크기가 ignore_iou_thresh 값보다 
                    클 경우에 이것은 학습에 사용되지 않도록 -1로 만들어준다.
                    -> 즉 박스는 3개의 셀 스케일당(13x13 26x26 52x52) 하나의 앵커박스를 정답으로 가지고 있도록 한다. 
    
                    '''

        return image, tuple(targets)



def test():
    anchors = config.ANCHORS
    transform = config.train_transforms

    dataset = YOLODataset(
        root=config.VAL_DIR,
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





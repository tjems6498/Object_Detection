import os
import cv2
import torch
import argparse
import numpy as np
import config
from model import YOLOv3
from util import cells_to_bboxes, non_max_suppression, show_image, my_non_max_suppression
import pdb
import time



parser = argparse.ArgumentParser()
parser.add_argument('--num-classes', type=int, default=11, help='')
parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
opt = parser.parse_args()

cap = cv2.VideoCapture(0)

torch.backends.cudnn.benchmark = True
model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)
checkpoint = torch.load('checkpoint.pth.tar', map_location=config.DEVICE)
model.load_state_dict(checkpoint['state_dict'])

model.eval()

colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(11)]
S = [13, 26, 52]
scaled_anchors = torch.tensor(config.ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)  # (3, 3, 2)
scaled_anchors = scaled_anchors.to(config.DEVICE)
while True:
    # time.sleep(1)
    ret, frame = cap.read()
    # image pre-processing before predict
    frame = cv2.resize(frame, (416, 416))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose(2,0,1)).to(config.DEVICE)  # 차원변경 + tensor + cuda
    img = img / 255.0  # scaling
    if img.ndimension() == 3:  # channels, width, height
        img = img.unsqueeze(0)  # 1 batch

    with torch.no_grad():
        output = model(img)

    boxes = []
    for i in range(output[0].shape[1]):  # y[0].shape : (batch, 3, 13, 13, 6)
        anchor = scaled_anchors[i]  # tensor(3, 2)
        # print(anchor.shape)
        # print(output[i].shape)
        boxes += cells_to_bboxes(
            output[i], is_preds=True, S=output[i].shape[2], anchors=anchor
        )[0]  # batch 제외 (num_anchors * S * S, 6)

    #boxes = non_max_suppression(boxes, iou_threshold=config.NMS_IOU_THRESH, threshold=config.CONF_THRESHOLD, box_format='midpoint')
    boxes = my_non_max_suppression(boxes, iou_threshold=0.3, threshold=config.CONF_THRESHOLD, score_threshold=0.3, box_format='midpoint', method='gaussian')

    print(len(boxes))
    # boxes : [[class_pred, prob_score, x1, y1, x2, y2], ...]

    image = show_image(frame, boxes, colors)

    cv2.imshow('fruit_detect', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 자원 반납
cv2.destroyAllWindows()
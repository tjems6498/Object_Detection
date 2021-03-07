import config
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import random
import torch
import cv2

from collections import Counter
from torch.utils.data import DataLoader
from tqdm import tqdm

import pdb


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[...,0], boxes2[..., 0]) * torch.min(boxes1[...,1], boxes2[..., 1])
    union = (boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection)

    return intersection / union


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    '''
    :param predictions: tensor of size (N, 3, S, S, num_classes+5)
    :param anchors: the anchors used for the predictions
    :param S: the number of cells the image is deivided in on the width and height
    :param is_preds: whether the input is predictions or the true bounding boxes
    :return: converted_bboxes: the converted boxes of sizes list(N, num_anchorsxSxS, 1+5) with class index,
                                object score, bounding box coordinates
    '''
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)  # 3
    box_predictions = predictions[..., 1:5]
    if is_preds:
        pass
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]


    cell_indices = (torch.arange(S)
                    .repeat(BATCH_SIZE,3,S,1)
                    .unsqueeze(-1)
                    .to(predictions.device))  # (Batch, 3, S, S, 1)  -> x 방향으로 0,1,2,3,4,5,6,7~
    # 중심을 가진 셀에 대해 0~1 값이었던 x,y를 현재 셀 스케일 전체에 대한 Normalize 좌표로 변환 (0~1)
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0,1,3,2,4))  # y방향으로 바꿔주고 더함
    w_h = 1 / S * box_predictions[..., 2:4]  # 절대좌표 -> Normalize
    converted_bboxes = torch.cat((best_class,scores,x,y,w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()



def non_max_suppression(bboxes, iou_threshold, threshold, box_format='corners'):
    '''

    :param bboxes: list of lists containing all bboxes with each bboxes
    specified as [class_pred, prob_score, x1, y1, x2, y2]
    :param iou_threshold: threshold where predicted bboxes is coorect
    :param threshold: threshold to remove predicted bboxes (independent of IoU)
    :param box_format: "midpoint" or "corners" used to specify bboxes
    :return:
        list: bboxes agter performing NMS given a specific IoU threshold
    '''

    assert type(bboxes) == list
    # input bboxes 값 확인하기
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes if box[0] != chosen_box[0] or intersection_over_union(
                    torch.tensor(box[2:]),
                    torch.tensor(chosen_box[2:]),
                    box_format=box_format) < iou_threshold]

        bboxes_after_nms.append(chosen_box)
    return bboxes_after_nms


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):

    # 코드 줄여보기
    if box_format == 'midpoint':
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    elif box_format == 'corners':
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def plot_image(image, boxes):
    cmap = plt.get_cmap('tab20b')
    class_labels = config.CLASSES
    colors = [cmap(i) for i in np.linspace(0,1,len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2

        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),  # 좌상단 Normalize 좌표를 원본 이미지 스케일로 변환
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    plt.show()






























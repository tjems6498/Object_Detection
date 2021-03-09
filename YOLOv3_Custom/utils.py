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
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)  # 1,3,1,1,2  broad casting
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])  # sigma(tx), sigma(tw)
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors  # e(tw) * pw
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)  # 아래서 concat하기 위해 unsqueeze 사용

    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]


    cell_indices = (torch.arange(S)
                    .repeat(BATCH_SIZE,3,S,1)
                    .unsqueeze(-1)
                    .to(predictions.device))  # (Batch, 3, S, S, 1)  -> x 방향으로 0,1,2,3,4,5,6,7~
    # 중심을 가진 셀에 대해 0~1 값이었던 x,y를 현재 셀 스케일 전체에 대한 Normalize 좌표로 변환 (0~1)
    # 거기에 +로 box prediction에서 Cx, Cy를 미리 더해놓음
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


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format='midpoint', num_classes=config.DEVICE
):
    """
    This function calculates mean average precision (mAP)
    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes
    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    average_precisions = []

    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)  # ap를 구할 클래스가 c인것만 따로 저장

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)


        # 각 example(batch만큼의 이미지들)이 가지고 있는 현재 클래스의 true box 개수
        amount_bboxes = Counter(gt[0] for gt in ground_truths)  # {0:3, 1:5}  인덱스:개수

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)  # amount_bboxes = {0:torch.tensor[0,0,0], 0:torch.tensor[0,0,0,0,0]}

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))   # True Positive (정답을 정답이라고 예측한 것)
        FP = torch.zeros((len(detections)))   # False Positive (배경을 정답이라고 예측한 것)
        total_true_bboxes = len(ground_truths)

        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):  # 여러 sample image 중에 한개의 detection box씩 뽑음
            # 우선 detection box의 이미지의 ground_truth를 가져옴
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)  # detection 한 이미지의 ground_truth 개수
            best_iou = 0

            # 예측 detection 박스와 그 이미지의 ground truth들과 다 비교하면서
            # iou가 가장 높은값이 그 물체를 예측하려 한 것임을 알 수 있음
            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            # 가장높은 iou값이 0.5보다 작다면 FP 크다면 TP
            if best_iou > iou_threshold:
                # ground truth는 한개의 detection값만 가질 수 있음
                # 즉 같은 한 물제에 두개의 예측 박스가 나오면 뒤에서 예측한 박스는 FP가 됨
                # o_p가 높은것부터 순서대로 진행이 되는데 o_p가 높다는 것은 그만큼 TP일 확률이 높기 때문에 납득 가능
                # amount_bboxes = {0:torch.tensor[0,0,0], 0:torch.tensor[0,0,0,0,0]}
                if amount_bboxes[detection[0]][best_gt_idx] == 0:  # 해당 gt를 처음 예측한 detection은 TP
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1  # iou가 높더라도 o_p가 더 높았던 이전 예측이 이미 예측한 gt이기 때문에 FP로 설정

            else:  # iou가 작다면 그냥 잘못된 예측 (background를 예측)
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)  # precision과 recall을 구하기 위해 누적합을 계산
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)  # TP / TP + FN   , epsilon
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        recalls = torch.cat((torch.tensor([0]), recalls))  # recall은 0부터 시작
        precisions = torch.cat((torch.tensor([1]), precisions))
        average_precisions.append(torch.trapz(precisions, recalls))  # trapz : y축, x축 을 주면 아래 면적 계산

    return sum(average_precisions) / len(average_precisions)



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


def get_evaluation_bboxes(
        loader,
        model,
        iou_threshold,
        anchors,
        threshold,
        box_format='midpoint',
        device='cuda',
):
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(tqdm(loader)):
        x = x.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]  # 각 배치리스트마다 3개의 스케일에 대한 예측값이 들어감
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S  # 앞에*을 붙여서 리스트를 벗고 들어감 -> tenor변환 하면서 차원하나 축소효과
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )  # 해당 스케일의 예측값을 t에서 b로 변환하고 (batch, SxSx3, 6)으로 차원변경
            for idx, (box) in enumerate(boxes_scale_i): # 배치만큼
                bboxes[idx] += box

        # 정답에 대한 변환값은 아무 스케일 1개에서 가져오면 됨
        true_bboxes = cells_to_bboxes(labels[2], anchor, S=S, is_preds=False)  # 52 x 52 스케일

        for idx in range(batch_size):  # 각 배치리스트 값마다 NMS를 수행
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )  # return list

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)  # 각 nms_box list에 0번째에 batch index 추가해서 append

            for box in true_bboxes:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes  # [[batch_idx, class_idx, o_p, x, y, w, h],[]...]



def check_class_accuracy(model, loader, threshold):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        if idx == 100:
            break
        x = x.to(config.DEVICE)
        with torch.no_grad():
            out = model(x)

        for i in range(3):
            y[i] = y[i].to(config.DEVICE)
            obj = y[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class/(tot_class_preds+1e-16))*100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj/(tot_noobj+1e-16))*100:2f}%")
    print(f"Obj accuracy is: {(correct_obj/(tot_obj+1e-16))*100:2f}%")
    model.train()





def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        'state_dict': model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)




def load_checkpoint(checkpoing_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoing_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # If we don't do this then it will just have learningrate of old checkpoint
    # and it will lead to many hours of debugging
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_loaders():
    '''
    test dataset, dataloader 만들어야함
    '''
    from dataset import YOLODataset

    IMAGE_SIZE = config.IMAGE_SIZE
    train_dataset = YOLODataset(
        root=config.ROOT,
        anchors=config.ANCHORS,
        S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        transform=config.train_transforms
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=True,
        drop_last=False,
    )

    train_eval_dataset = YOLODataset(
        root=config.ROOT,
        anchors=config.ANCHORS,
        S=[IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8],
        transform=config.test_transforms
    )
    train_eval_loader = DataLoader(
        dataset=train_eval_dataset,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, train_eval_loader











def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




'''
class loss는 데이터에따라  bce를 쓸지 안쓸지 정함

'''


import random
import torch
import torch.nn as nn

from util import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()  # logistic + CrossEntropy  -> multi label classification
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()


        # Constants signifying how much to pay for each respectivve part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0


        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj])
        )


        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)  # w와 h를 가진 3개의 anchor가 모든 셀에서 계산하기위해 broad casting을 사용
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]),torch.exp(predictions[..., 3:5])*anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.bce((predictions[..., 0:1][obj]), (ious * target[..., 0:1][obj]))
        # 내 생각 : true box와 anchor box의 iou를 계산해서 그것을 예측할 수 있도록 loss를 구성
        # 즉 object prob * IOU = confidence 가 0.8이고 에측값의 object prob가 0.4이면 0.4가 아닌 0.8이 되도록 학습을 진행시킴


        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates  # target을 bw,bh에서 tw, wh상태로 만들어줌
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])


        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[...,5:][obj]), (target[..., 5][obj].long())
        )


        #    print("__________________________________")
        #    print(self.lambda_box * box_loss)
        #    print(self.lambda_obj * object_loss)
        #    print(self.lambda_noobj * no_object_loss)
        #    print(self.lambda_class * class_loss)
        #    print("\n")

        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )











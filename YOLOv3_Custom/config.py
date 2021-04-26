import albumentations as A
import cv2
import torch

from albumentations.pytorch import ToTensorV2
from util import seed_everything

DATASET = 'PASCAL_VOC'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seed_everything()  # deterministic behavior
NUM_WORKERS = 2  # colab
BATCH_SIZE = 3
IMAGE_SIZE = 416
NUM_CLASSES = 11
CLASSES = ['apple', 'orange', 'pear', 'watermelon', 'durian', 'lemon', 'grapes', 'pineapple', 'dragon fruit', 'oriental melon', 'melon']
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.6
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
PIN_MEMORY = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth2.tar"
TRAIN_DIR = 'E:\\Computer Vision\\data\\project\\fruit_yolov3\\train'
VAL_DIR = 'E:\\Computer Vision\\data\\project\\fruit_yolov3\\valid'

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]  # Note these have been rescaled to be between [0, 1]

scale = 1.1
train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),   # 초기 이미지의 비율을 유지하면서 한쪽(w,h)이 max_size와 같도록 이미지 크기 조정 (가로 세로중 한쪽이 416*1.1 이 되도록 resize)
        A.PadIfNeeded(  # 입력 이미지 size가 min_height, min_width값이 될때 까지 0으로 채움
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.RandomBrightnessContrast(p=0.2),  # new
        # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=10, p=0.4, border_mode=cv2.BORDER_CONSTANT
                ),
                A.IAAAffine(shear=10, p=0.4, mode="constant"),  # rotate와 비슷한 느낌
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Blur(p=0.05),
        A.CLAHE(p=0.1),  # 이미지가 뭔가 진해지고 선명해짐 / Doc: Apply Contrast Limited Adaptive Histogram Equalization
        # A.Posterize(p=0.1),
        # A.ToGray(p=0.1),

        A.Normalize(mean=[0.6340, 0.5614, 0.4288], std=[0.2803, 0.2786, 0.3126], max_pixel_value=255,),
        ToTensorV2(),

    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),  # 후의 박스 면적이 전의 면적의 0.4 이하이면 사용x
)
test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT
        ),
        A.Normalize(mean=[0.6340, 0.5614, 0.4288], std=[0.2803, 0.2786, 0.3126], max_pixel_value=255,),  # 이미지 Normalize
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
)








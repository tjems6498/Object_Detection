import cv2
import torch
import argparse
import config
from model import YOLOv3


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-classes', type=int, default=11, help='')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--load-model', type=bool, default=True, help='load trained model')
    opt = parser.parse_args()

    cap = cv2.VideoCapture(0)

    torch.backends.cudnn.benchmark = True
    model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    if opt.load_model:
        checkpoint = torch.load('checkpoint.pth.tar',map_location=config.DEVICE)
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    while True:
        ret, frame = cap.read()
        img = torch.from_numpy(frame).to(config.DEVICE)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 1 batch

        output = model(img)
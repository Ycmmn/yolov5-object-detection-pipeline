import torch
import torchvision
import cv2    # for opencv-python
import numpy as np

import sys
import time 
from collections import Counter

cv2.setUseOptimized(True)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cudnn.benchmark = True      # increase FPS speed

# the labels that we want to be in YOLO
WANT = {'person','car','truck','bus','motorcycle','bicycle','dog','cat','bird','horse','cow','sheep'}


# function to load and return the YOLOv5 model 
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(DEVICE)
    model.conf = 0.35     # Confidence threshold
    model.iou = 0.45     # threshold for bounding box
    model.max_det = 300    # maximum number
    
    if DEVICE == 'cuda':
        model.half()    # FP16
    return model


    


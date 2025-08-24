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

def load_model():
    # load the model and set it up simply
    modelmodel = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(DEVICE)


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
    
    # get the class names that the model can detect
    names = model.names  # we dont know dose it dict or list

    #find number of classes that we want
    if isinstance(names, dict):
        # If names is a dict
        model.classes = [i for i,n in names.items() if n in WANT ]

    else:
        #If names is a list
        model.classes = [i for i,n in enumerate(names) if n in WANT]
    
    return model 


def open_source(src):
    # open video source with OpenCV
        cap = cv2.VideoCapture(src)

        #  if  video source not opened, stop the program with an error message
        if not cap.isOpened():
            raise SystemExit("❌ Video source could not be opened.")

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # use small buffer for less delay 

        return cap 
            


# warm up the model
def warmup(model):
    # create a fake input (zero tensor) with shape (batch=1, channels=3, height=640, width=640)
    x = torch.zeros((1, 3, 640, 640), device=DEVICE, dtype=torch.float16 if DEVICE == 'cuda' else torch.float32)

    # no need to compute gradients during warmup
    with torch.no_grad():
        _ = model(x)   # run once to get the model ready and faster 


# read one frame from video source  
def get_frame(cap):
    # try to read one frame from source
    ret , frame = cap.read()  # if ret is True ==> return frame
    # if success return the frame else stop the loop
    return frame if ret else None


# YOLOv5 handles preprocessing==>  raw frame is enough
# for future use if we need preprocessing in models without built-in support
def preprocess(frame):
    return frame

    

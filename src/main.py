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


@torch.no_grad()     # disable gradient for all this code block (inference only)
def infer(model, image, size=640):
    model.eval()  #set evaluation mode
    results = model(image, size=size)  # run model on input
    return results



def postprocess(results, names):
    # get predictions
    pred = getattr(results, 'pred', None)
    pred = pred[0] if pred is not None else results.xyxy[0]

    # if no prediction, return empty
    if pred is None or len(pred) == 0:
        return [], Counter()

    # get class ids
    cls_ids = pred[:, -1].int().tolist()

    # set ids to labels
    labels = [names[c] for c in cls_ids]

    # return labels and count
    return labels, Counter(labels)



# the goal is to draw boxes on the frame and write a header on top of it
# the header shows FPS and the counts of detected objects
def draw_and_compose(result, counts, fps):
    # draw boxes on the frame
    vis = result.render()[0]

    # create text for FPS
    fps_text = f"FPS: {fps:.1f} | "

    # create text for object counts
    if counts:  # counts is not None
        counts_text = " | ".join(f"{k}:{v}" for k, v in counts.items())

    else:  # counts is None
        counts_text = "---"

    # combine both texts
    head = fps_text + counts_text

    font = cv2.FONT_HERSHEY_SIMPLEX 

    # write black shadow text at the top of the frame
    cv2.putText(vis, head, (10, 26), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # write white text on top (for contrast)
    cv2.putText(vis, head, (10, 26), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # return the frame with bounding boxes and header
    return vis




# To check whether the client needs to save video or not
def maybe_open_writer(save_flag, writer, frame_like, out_path='output.mp4', fps=30):

    # if x:    ---> x is True
    # if not x: ---> x is False
    if not save_flag or writer is not None:
        return writer
    
    # get frame height and width
    h, w = frame_like.shape[:2]

    # open video writer
    return cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))




def write_and_show(writer, vis):
    if writer: writer.write(vis)
    cv2.imshow("Object Detection (q to quit)", vis)
 
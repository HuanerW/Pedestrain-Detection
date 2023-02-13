from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2
from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2
import csv
# frame number , csv file 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import requests
dataset=[]
link = "https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names"
classes = requests.get(link)
names = classes.text.split("\n")[:-1]
for i in range (11):
    names.append('nothing');

names[1]='person'
dataset.append(names[1])
COLORS = np.random.uniform(0, 255, size=(len(names), 3))

model = detection.retinanet_resnet50_fpn(pretrained=True, progress=True,
num_classes=len(names), pretrained_backbone=True).to(DEVICE)
model.eval()
cap=cv2.VideoCapture('/Users/jules/Downloads/KIDS.mp4')
outList=[]
sX=200
sY=108
filename='/Users/jules/downloads/index.csv'
k=0
while len(outList)<=10:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = cap.read();
    frame = imutils.resize(frame, width=400)
    orig = frame.copy()
    # convert the frame from BGR to RGB channel ordering and change
    # the frame from channels last to channels first ordering
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.transpose((2, 0, 1))
    # add a batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the frame to a floating point tensor
    frame = np.expand_dims(frame, axis=0)
    frame = frame / 255.0
    frame = torch.FloatTensor(frame)
    # send the input to the device and pass the it through the
    # network to get the detections and predictions
    frame = frame.to(DEVICE)
    k+=1;
    detections = model(frame)[0]
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]
        if confidence > 0.5:
            idx = int(detections["labels"][i])
            if(names[idx]=='person'):
                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                print("edX and startX is "+ str(endX)+"and"+str(startX))
                print("This is sX"+str(sX))
                if(abs((endX+startX)/2-sX)<=10 and abs((endY+startY)/2-sY)<=10):
                   label = "{}: {:.2f}%".format(names[idx], confidence * 100)
                   cv2.rectangle(orig, (startX, startY), (endX, endY),COLORS[idx], 2)
                   y = startY - 15 if startY - 15 > 15 else startY + 15
                   outList.append([k,(endX+startX)/2,(startY+endY)/2])
                   print(outList)
                   sX=(endX+startX)/2
                   sY=(startY+endY)/2
                   cv2.putText(orig, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                   
             
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(30) & 0xFF
    cv2.destroyAllWindows()
with open(filename,'w') as f:
   write=csv.writer(f)
   write.writerow({'VideoX','Frame','VideoY'})
   write.writerows(outList)


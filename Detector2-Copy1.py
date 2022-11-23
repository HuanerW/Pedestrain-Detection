#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchvision.models import detection
import numpy as np
import argparse
import pickle
import torch
import cv2


# In[2]:


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:





# In[ ]:





# In[3]:


import requests
dataset=[]
link = "https://raw.githubusercontent.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch/master/data/coco.names"
classes = requests.get(link)
names = classes.text.split("\n")[:-1]
for i in range (11):
    names.append('nothing');

names[1]='person'
dataset.append(names[1])
print(dataset)


# In[4]:


model = detection.retinanet_resnet50_fpn(pretrained=True, progress=True,
num_classes=len(names), pretrained_backbone=True).to(DEVICE)
model.eval()


# In[5]:


image = cv2.imread('/Users/jules/Downloads/FUN.jpeg')
orig = image.copy()
COLORS = np.random.uniform(0, 255, size=(len(names), 3))


# In[6]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))


# In[16]:


image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)


# In[17]:


image = image.to(DEVICE)
detections = model(image)[0]


# In[18]:


for i in range(0, len(detections["boxes"])):
# extract the confidence (i.e., probability) associated with the
# prediction
    confidence = detections["scores"][i]
# filter out weak detections by ensuring the confidence is
# greater than the minimum confidence
    if confidence > 0.5:
# extract the index of the class label from the detections,# then compute the (x, y)-coordinates of the bounding box# for the object
        
        idx = int(detections["labels"][i])
        box = detections["boxes"][i].detach().cpu().numpy()
        if(names[idx]=='person'):
              #  box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int") # display the prediction to our terminal
                label = "{}: {:.2f}%".format(names[idx], confidence * 100)
                print("[INFO] {}".format(label)) # draw the bounding box and label on the image
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)


# In[ ]:



cv2.imshow("Output", orig)
cv2.waitKey(0)


# In[18]:





# In[7]:


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


# In[8]:


model = detection.retinanet_resnet50_fpn(pretrained=True, progress=True,
num_classes=len(names), pretrained_backbone=True).to(DEVICE)
model.eval()


# In[9]:


cap=cv2.VideoCapture('/Users/jules/Downloads/KIDS.mp4')


# In[10]:


while True:
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
    detections = model(frame)[0]
    for i in range(0, len(detections["boxes"])):
        confidence = detections["scores"][i]
        if confidence > 0.5:
            idx = int(detections["labels"][i])
            if(names[idx]=='person'):
                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                label = "{}: {:.2f}%".format(names[idx], confidence * 100)
                cv2.rectangle(orig, (startX, startY), (endX, endY),COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(30) & 0xFF
    cv2.destroyAllWindows()


# In[2]:





# In[3]:





# In[ ]:





# In[ ]:





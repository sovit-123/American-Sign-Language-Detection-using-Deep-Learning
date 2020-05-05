'''
USAGE:
python test.py 
'''

import torch
import joblib
import torch.nn as nn
import numpy as np
import cv2
import argparse
import pretrainedmodels
import torch.nn.functional as F
import time
import cnn_models
 
from torchvision import models

# load label binarizer
lb = joblib.load('../outputs/lb.pkl')

# class CustomCNN(nn.Module):
#     def __init__(self):
#         super(CustomCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(16, 32, 5)
#         self.fc1 = nn.Linear(32, 64)
#         self.fc2 = nn.Linear(64, 128)
#         self.fc3 = nn.Linear(128, len(lb.classes_))

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         bs, _, _, _ = x.shape
#         x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
model = cnn_models.CustomCNN().cuda()
model.load_state_dict(torch.load('../outputs/model.pth'))
print(model)
print('Model loaded')

def findHand(img):
    hand = img[100:324,100:324]
    hand = cv2.resize(hand, (224,224))
    # hand = hand/255
    return hand

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print('Error while trying to open camera. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object
out = cv2.VideoWriter('out_videos/asl.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (100, 100), (324, 324), (20,34,255), 2)
    hand = findHand(frame)

    image = hand
    
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image = torch.tensor(image, dtype=torch.float).cuda()
    image = image.unsqueeze(0)
    
    outputs = model(image)
    _, preds = torch.max(outputs.data, 1)
    # print('PREDS', preds)
    # print(f"Predicted output: {lb.classes_[preds]}")
    
    cv2.putText(frame, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imshow('image', frame)
    # cv2.imwrite(f"../outputs/{args['img']}", frame)

    # time.sleep(0.09)

    # press `q` to exit
    if cv2.waitKey(27) & 0xFF == ord('q'):
        break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()
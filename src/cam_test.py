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
import albumentations
import pretrainedmodels
import torch.nn.functional as F
import time
 
from torchvision import models

# construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--img', default='A_test.jpg', type=str,
    help='path for the image to test on')
args = vars(parser.parse_args())

# load label binarizer
lb = joblib.load('../outputs/lb.pkl')

aug = albumentations.Compose([
                albumentations.Resize(224, 224, always_apply=True),
])

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, len(lb.classes_))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
model = CustomCNN().cuda()

model.load_state_dict(torch.load('../outputs/model.pth'))
print('Model loaded')

cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print('Error while trying to open camera. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# define codec and create VideoWriter object
out = cv2.VideoWriter('out_videos/cam_blur.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # add gaussian blurring to frame

        image = frame
        image_copy = image.copy()
        
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).cuda()
        image = image.unsqueeze(0)
        # print(image.shape)
        
        outputs = model(image)
        _, preds = torch.max(outputs.data, 1)
        # print('PREDS', preds)
        # print(f"Predicted output: {lb.classes_[preds]}")
        
        cv2.putText(image_copy, lb.classes_[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.imshow('image', image_copy)
        cv2.imwrite(f"../outputs/{args['img']}", image_copy)

        time.sleep(0.05)

        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()
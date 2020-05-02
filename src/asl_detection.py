import torch
import random
import cv2 as cv
import numpy as np
import os

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

'''
SEED Everything
'''
def seed_everything(SEED=42):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True # as we will resize the images
SEED=42
seed_everything(SEED=SEED)
'''
SEED Everything
'''

device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

# define transforms
train_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
val_transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])

# get the paths
image_paths = list(paths.list_images('../input/asl_alphabet_train/asl_alphabet_train'))

data = []
labels = []
for image_path in tqdm(image_paths, total=len(image_paths)):
    label = image_path.split(os.path.sep)[-2]

    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = cv.resize(image, (224, 224))

    # data.append(image)
    labels.append(label)

data = np.array(data)
labels = np.array(labels)

print(f"Total images: {len(data)}")
print(f"Total labels: {len(labels)}")

print('####################')

print('First 10 labels')
for i in range(10):
    print(f"CATEGORY {i}: {labels[i]}")
	
print('####################')

# one hot encode
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
print('After one hot encoding')
for i in range(10):
    print(labels[i])

print('####################')

print('First 10 labels')
for i in range(10):
    print(f"LABEL {i}: {lb.classes_[i]}")

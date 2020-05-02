import pandas as pd
import os
import cv2
import random
import albumentations
import numpy as np

from imutils import paths
from tqdm import tqdm

# get all the image paths
image_paths = list(paths.list_images('../input/asl_alphabet_train/asl_alphabet_train'))
dir_paths = os.listdir('../input/asl_alphabet_train/asl_alphabet_train')
dir_paths.sort()

root_path = '../input/asl_alphabet_train/asl_alphabet_train'

# get 1000 images from each category
for idx, dir_path  in tqdm(enumerate(dir_paths), total=len(dir_paths)):
    all_images = os.listdir(f"{root_path}/{dir_path}")
    os.makedirs(f"../input/preprocessed_image/{dir_path}", exist_ok=True)
    for i in range(1000): # how many images to preprocess
        # generate a random id between 0 and 2999
        rand_id = (random.randint(0, 2999))
        image = cv2.imread(f"{root_path}/{dir_path}/{all_images[rand_id]}")
        image = cv2.resize(image, (224, 224))

        cv2.imwrite(f"../input/preprocessed_image/{dir_path}/{dir_path}{i}.jpg", image)
from augmentations import aug_ops, AugmentImage
from glob import glob
from tqdm import tqdm
import sys
import os
import cv2
import random
import json
import numpy as np
import pathlib
from PIL import Image
import string
import multiprocessing
from tqdm.contrib.concurrent import process_map

config = json.loads(open('config.json').read())

background_image_dir = config['background_dir']
gs_image_dir = config['green_screen_dir']
num_augments = config['augment_per_image']
size = config['size'], config['size']
output_dir = sys.argv[1]
background_images = glob(os.path.join(background_image_dir, '**', '*.png'), recursive=True) + glob(
        os.path.join(background_image_dir, '**', '*.jpg'), recursive=True)

label2image = {}
lock = multiprocessing.Lock()

files = glob(os.path.join(gs_image_dir, 'hand_images', '*'))
files = list(set(files))
labels = [int(f.split(os.sep)[-1])for f in files]
last_label = max(labels)


def create_background_images(label):
    bg_image = random.choice(background_images)
    orig_bg_img = cv2.imread(bg_image)
    for n in range(num_augments):
        aug_bg = AugmentImage(img=orig_bg_img, mask=None)
        bg_img, _ = aug_bg.augment_image()
        bg_img = cv2.resize(bg_img, (224, 224))
        img = Image.fromarray(bg_img)
        aug = np.random.choice(aug_ops)
        img = aug(img)
        img = np.array(img, dtype=np.uint8)
        pathlib.Path(os.path.join(output_dir, str(label+1))).mkdir(parents=True, exist_ok=True)
        rand_filename = ''.join(random.choices(string.ascii_lowercase, k=7))+'.png'
        cv2.imwrite(os.path.join(output_dir, str(label+1),  rand_filename), img)


if __name__ == '__main__':
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    from tqdm.contrib.concurrent import process_map
    process_map(create_background_images, [last_label]*1000, max_workers=10, chunksize=1)
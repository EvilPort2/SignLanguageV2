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


def create_image(bg_img, fg_img, mask):
    # img = np.random.random((size, size, 3)).astype(np.uint8)
    #print(bg_img.shape)
    bg_img_resized = cv2.resize(bg_img, size, cv2.INTER_AREA)
    fg_img_resized = cv2.resize(fg_img, size, cv2.INTER_AREA)
    mask_img_resized = cv2.resize(mask, size, cv2.INTER_AREA)
    _, mask_img_resized = cv2.threshold(mask_img_resized, 50, 255, cv2.THRESH_BINARY)
    fg_img_resized = cv2.bitwise_and(fg_img_resized, fg_img_resized, mask=mask_img_resized[:,:,0])
    img = cv2.bitwise_and(bg_img_resized, 255-mask_img_resized, mask=255-mask_img_resized[:,:,0]) + fg_img_resized
    return img


def create_hand_image(gs_image):
    mask_image = gs_image.replace('green_screen', 'mask')
    label = gs_image.replace(os.path.join(gs_image_dir, 'green_screen'), '').strip(os.sep).split(os.sep)[0]
    hand_dir = os.path.join(output_dir, label)
    orig_mask_img = cv2.imread(mask_image)
    orig_fg_img = cv2.imread(gs_image)
    bg_image = random.choice(background_images)
    orig_bg_img = cv2.imread(bg_image)
    for n in range(num_augments):
        aug_bg = AugmentImage(img=orig_bg_img, mask=None)
        bg_img, _ = aug_bg.augment_image()
        img = create_image(bg_img, orig_fg_img, orig_mask_img)
        img = Image.fromarray(img)
        aug = np.random.choice(aug_ops)
        mag = np.random.choice([0, 1, 2, 3, 4, ])
        img = aug(img, mag=mag)
        img = np.array(img, dtype=np.uint8)
        pathlib.Path(hand_dir).mkdir(parents=True, exist_ok=True)
        rand_filename = ''.join(random.choices(string.ascii_lowercase, k=7))+'.png'
        cv2.imwrite(os.path.join(output_dir, label,  rand_filename), img)


if __name__ == '__main__':
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(os.path.join(background_image_dir, '**', '*.png'))
    gs_images = glob(os.path.join(gs_image_dir, 'green_screen', '**', '*.png'), recursive=True)
    gs_images.sort()
    filtered_gs_images = []
    for gs_image in tqdm(gs_images):
        label = gs_image.replace(os.path.join(gs_image_dir, 'green_screen'), '').strip(os.sep).split(os.sep)[0]
        hand_dir = os.path.join(output_dir, label)
        if not os.path.exists(hand_dir):
            filtered_gs_images.append(gs_image)
    # max_processes = 10
    # for i in tqdm(range(0, len(gs_images), max_processes)):
    #     images = gs_images[i:i+max_processes]
    #     processes = []
    #     for image in images:
    #         p = multiprocessing.Process(target=create_hand_image, args=(image, ))
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()

    process_map(create_hand_image, filtered_gs_images, max_workers=10, chunksize=1)
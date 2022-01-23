import cv2
import os
import pathlib
from glob import glob
import sys
from tqdm import tqdm

data_dir = sys.argv[1]
mask_out_dir = sys.argv[2]

files = glob(os.path.join(data_dir, '**/*.png'), recursive=True)

hsv_range = (17, 0, 0, 96, 255, 255)

for image in tqdm(files):
    img = cv2.imread(image)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, hsv_range[:3], hsv_range[3:])
    mask = 255 - mask
    mask_image_file = image.replace(data_dir, mask_out_dir)
    pathlib.Path(os.path.dirname(mask_image_file)).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(mask_image_file, mask)
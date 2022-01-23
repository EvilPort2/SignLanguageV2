from glob import glob
import os
import sys
import random
import pandas as pd
import argparse


parser = argparse.ArgumentParser('Create dataset csv files')
parser.add_argument('--img_dir', help='Image directory that contains the hand images')
parser.add_argument('--save_path', help='Directory to save the csv files')
args = parser.parse_args()


dir_path = args.img_dir
save_path = args.save_path
random.seed(2021)

files = glob(os.path.join(dir_path, '*', '*.png'))
labels = []
for image in files:
    image = image.replace(dir_path, '')
    image = image.strip(os.sep)
    label = int(image.split(os.sep)[0])
    labels.append(label)

df = pd.DataFrame({
    'image': files,
    'label': labels
})

df = df.sample(frac=1)
train_df = df[:int(len(df)*0.6)]
test_df = df[int(len(df)*0.6): int(len(df)*0.8)]
val_df = df[int(len(df)*0.8):]

train_df.to_csv(os.path.join(save_path, 'train.csv'), index=None)
test_df.to_csv(os.path.join(save_path, 'test.csv'), index=None)
val_df.to_csv(os.path.join(save_path, 'val.csv'), index=None)

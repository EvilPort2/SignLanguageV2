import cv2
import sys
import pathlib
import os
sys.path.append('..')
from live_prediction.predictor import Predictor
from PIL import Image
import numpy as np
import argparse


parser = argparse.ArgumentParser('Live Prediction')
parser.add_argument('--model_dir', help='Model directory')
args = parser.parse_args()


model_dir = args.model_dir


predictor = Predictor(model_dir)

cam = cv2.VideoCapture(0)
size = 200
x, y = 400, 130
hand_box = [x, y, x+size, y+size]
is_capturing = False
frame_num = 0
while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (hand_box[0], hand_box[1]), (hand_box[2], hand_box[3]), (0, 255, 0), 1)
    cv2.imshow('cam', frame)
    hand_image = frame[hand_box[1]:hand_box[3], hand_box[0]: hand_box[2]]
    cv2.imshow('hand', hand_image)
    wait_key = cv2.waitKey(1)
    if wait_key == ord('q'):
        print(frame.shape)
        break
    hand_img = Image.fromarray(hand_image)
    label = predictor(hand_img)
    blackboard = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.putText(blackboard, str(label), (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    cv2.imshow('blackboard', blackboard)

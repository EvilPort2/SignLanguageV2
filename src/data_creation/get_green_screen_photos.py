import cv2
import sys
import pathlib
import os


store_photo = sys.argv[1]
num_photos = int(sys.argv[2])

pathlib.Path(store_photo).mkdir(parents=True, exist_ok=True)

cam = cv2.VideoCapture(0)
size = 200
x, y = 400, 130
hand_box = [x, y, x+size, y+size]
is_capturing = False
frame_num = 0
while True and num_photos > frame_num:
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
    if wait_key == ord('c'):
        if not is_capturing:
            is_capturing = True
        else:
            is_capturing = False
    if is_capturing:
        cv2.imwrite(os.path.join(store_photo, str(frame_num)+'.png'), hand_image)
        frame_num += 1
        print(frame_num, end='\r')
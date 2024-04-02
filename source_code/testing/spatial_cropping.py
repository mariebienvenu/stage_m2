
import os
from time import time

import numpy as np
import cv2

from app.Video import Video

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

VIDEO_NAME = '03-28 used red object/sec' #21 added light and glove/close_startup'
video = Video(data_path+VIDEO_NAME+'.mp4')
i = 0
close_key, previous_key, next_key = 'q', 'b', 'n' #quit, before, next

x1, x2, y1, y2 = 0, 0, 0, 0
drawing = False
frame = video.get_frame(i)

def draw_rectangle(event,x,y,flags,param):
    global i, x1, x2, y1, y2, drawing, frame

    frame = video.get_frame(i)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x,y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            x2, y2 = x,y
            #cv2.rectangle(frame,(x1, y1),(x,y),(0,255,0),-1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x2, y2 = x,y
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),-1)

cv2.namedWindow(VIDEO_NAME)
cv2.setMouseCallback(VIDEO_NAME, draw_rectangle)

while True:
    current = max(0, min(i, video.frame_count))
    frame = video.get_frame(i)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
    cv2.imshow(VIDEO_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(close_key):
        cv2.destroyWindow(VIDEO_NAME)
        break
    elif key == ord(previous_key) and i>=1:
        i -= 1
    elif key == ord(next_key) and i<video.frame_count-1:
        i += 1

    print(f'Displaying frame {i}')

print(f"Crop zone chosen : x1={x1}, x2={x2}, y1={y1}, y2={y2}")

crop = np.array([[x1, y1], [x2, y2]])
np.save(data_path+VIDEO_NAME+'_crop.npy', crop)
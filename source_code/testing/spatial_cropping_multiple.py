
import os
from time import time

import numpy as np
import cv2

from app.video import Video
from app.color import Color

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

VIDEO_NAME = '03-28 used red object/sec' #21 added light and glove/close_startup'
video = Video(data_path+VIDEO_NAME+'.mp4')
i = 0
close_key, previous_key, next_key = 'q', 'b', 'n' #quit, before, next
add_key, delete_key = 'a', 'd'
change_current_left_key, change_current_right_key = 'i', 'o'


rectangles = [
    {"x1":0, "x2":0, "y1":0, "y2":0}
]
current_rectangle_index=0

drawing = False
frame = video.get_frame(i)

def draw_rectangle(event,x,y,flags,param):
    global i, rectangles, current_rectangle_index, drawing, frame

    frame = video.get_frame(i)
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        rectangles[current_rectangle_index]["x1"], rectangles[current_rectangle_index]["y1"] = x,y
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            rectangles[current_rectangle_index]["x2"], rectangles[current_rectangle_index]["y2"] = x,y
            #cv2.rectangle(frame,(x1, y1),(x,y),(0,255,0),-1)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rectangles[current_rectangle_index]["x2"], rectangles[current_rectangle_index]["y2"] = x,y
        #cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),-1)

cv2.namedWindow(VIDEO_NAME)
cv2.setMouseCallback(VIDEO_NAME, draw_rectangle)

while True:
    Color.reset()
    current = max(0, min(i, video.frame_count))
    frame = video.get_frame(i)
    for rect in rectangles:
        x1, x2, y1, y2 = rect.values()
        color = Color.next()
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
    cv2.imshow(VIDEO_NAME, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(close_key):
        cv2.destroyWindow(VIDEO_NAME)
        break
    elif key == ord(previous_key) and i>=1:
        i -= 1
    elif key == ord(next_key) and i<video.frame_count-1:
        i += 1
    elif key == ord(add_key):
        current_rectangle_index = len(rectangles)
        rectangles.append({"x1":0, "x2":0, "y1":0, "y2":0})
    elif key == ord(delete_key):
        rectangles.pop(current_rectangle_index)
        current_rectangle_index -= 1
    elif key == ord(change_current_left_key):
        if current_rectangle_index > 0:
            current_rectangle_index -= 1
    elif key == ord(change_current_right_key):
        if current_rectangle_index < len(rectangles)-1:
            current_rectangle_index += 1

    print(f'Displaying frame {i} and modifying rectangle {current_rectangle_index}/{len(rectangles)}')

print(f"Crop zones chosen : {rectangles}")
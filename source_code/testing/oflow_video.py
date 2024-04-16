import os
from time import time

import numpy as np
import cv2
from tqdm import tqdm

from app.Video import Video
import app.OpticalFlow as OpticalFlow

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

VIDEO_NAME = '03-28 used red object/sec' #21 added light and glove/close_startup'
video = Video(data_path + VIDEO_NAME +'.mp4', verbose=1)

oflow_len = video.frame_count-1

oflow_video_content = np.zeros((oflow_len, video.frame_height, video.frame_width, 3), dtype=np.uint8)

for index in tqdm(range(oflow_len), desc='Oflow computation'):
    t0 = time()
    flow = video.get_optical_flow(index, degrees=False) #no filtering
    bgr = flow.make_oflow_image()
    oflow_video_content[index,:,:,:] = np.copy(bgr)

    cv2.imshow("oflow", bgr)
    t1 = time()
    delay = max(1, int(video.wait_time-(t1-t0)))
    key = cv2.waitKey(delay)
    if key == ord('q'):
        break

#oflow_video1 = Video.from_array(oflow_video_content, data_path+f'/{VIDEO_NAME}_oflow.avi', fps=video.fps)
oflow_video2 = Video.from_array(oflow_video_content, data_path+f'/{VIDEO_NAME}_oflow.mp4', fps=video.fps)
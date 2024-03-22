import os

import cv2
import numpy as np

from app.Video import Video
import app.optical_flow as optical_flow

data_path = 'C:/Users/Marie Bienvenu/WORKSPACE/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

VIDEO_NAME = '03-21 added light and glove/close_startup'
video = Video(data_path + VIDEO_NAME +'.mp4', verbose=1)

oflows = np.zeros((video.frame_count, video.frame_height, video.frame_width, 2), dtype=np.float64)
oflow_video_content = np.zeros((video.frame_count, video.frame_height, video.frame_width, 3), dtype=np.uint8)

for index in range(video.frame_count-1):
    frame1 = cv2.cvtColor(video.get_frame(index), cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(video.get_frame(index+1), cv2.COLOR_BGR2GRAY)

    oflow = optical_flow.compute_oflow(frame1, frame2, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2)
    oflows[index,:,:,:] = oflow

    magnitude, angle = optical_flow.cartesian_to_polar(oflow)
    mask = optical_flow.get_mask_oflow(magnitude)
    oflow *= mask
    new_mag, new_angle = optical_flow.cartesian_to_polar(oflow)
    bgr = optical_flow.make_oflow_image(new_mag, new_angle)
    oflow_video_content[index,:,:,:] = np.copy(bgr)

    cv2.imshow("oflow", bgr)
    key = cv2.waitKey(video.wait_time)
    if key == ord('q'):
        break

oflow_video1 = Video.from_array(oflow_video_content, data_path+f'/{VIDEO_NAME}_oflow.avi', fps=video.fps)
oflow_video2 = Video.from_array(oflow_video_content, data_path+f'/{VIDEO_NAME}_oflow.mp4', fps=video.fps)
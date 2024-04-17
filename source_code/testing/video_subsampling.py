
import os
import numpy as np
import cv2

import app.Video as Video

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

subdirectory = '04-08 appareil de Damien' #'03-11 initial videos'
VIDEO_NAME = 'P1010236'

video = Video.Video(f'{data_path}/{subdirectory}/{VIDEO_NAME}.mp4', verbose=1)
video.load(verbose=1)

subsampled = video.subsample(rate=3, verbose=1)
same_appearance = video.subsample(rate=6, fps="KEEP", verbose=1)


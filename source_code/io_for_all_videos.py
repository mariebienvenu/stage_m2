

import os, sys

from time import time

t0 = time()

from app.video_io import VideoIO, default_config
from app.color import Color

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

directories = os.listdir(data_path)

time_in_seconds = False

for directory in directories:

    try:
        filenames = os.listdir(f'{data_path}/{directory}')
    except NotADirectoryError: # same as checking for os.path.isdir(directory)
        filenames = []

    for filename in filenames:

        if (".mp4" in filename or ".MP4" in filename) and filename[0]!="_": # is a video and is not a working file

            video_name = filename[:-4]
            extension = filename[-4:]
            print(f'Currently processing video "{video_name}".')
            video_io = VideoIO(f'{data_path}/{directory}/', video_name, extension=extension, verbose=10)
            video_io.save_config()

            if (
                len(video_io.spatial_crops)==1 and
                video_io.spatial_crops[0]['x1']==0 and 
                video_io.spatial_crops[0]['y1']==0 and
                video_io.spatial_crops[0]['x2']==video_io.video.frame_width and 
                video_io.spatial_crops[0]['y2']==video_io.video.frame_height
            ):
                video_io.get_spatial_crop_input_from_user()

            '''if '404' in video_name or '405' in video_name:
                video_io.get_spatial_crop_input_from_user()'''
            
            video_io.process()
            if video_io.time_crop[0]==0 and video_io.time_crop[1]==video_io.oflow_len:
                video_io.auto_time_crop()      

            video_io.draw_diagrams(time_in_seconds=time_in_seconds)
            video_io.to_animation()

            '''if '404' in video_name or '405' in video_name:
                vid = video_io.record_video() # a bit long + requires loading the entire video in memory...'''

tf = time()
print(f'Computation took {int(tf-t0)} seconds.') # less than two minutes for 10 videos # way longer if longer videos or higher fps -> up to 10 minutes per video

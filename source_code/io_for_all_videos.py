

import os, sys

from app.VideoIO import VideoIO, default_config

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

directories = os.listdir(data_path)

for directory in directories:

    try:
        filenames = os.listdir(f'{data_path}/{directory}')
    except NotADirectoryError: # same as checking for os.path.isdir(directory)
        filenames = []

    for filename in filenames:

        if ".mp4" in filename:

            video_name = filename[:-4]
            video_io = VideoIO(f'{data_path}/{directory}/', video_name, verbose=10)

            if (
                video_io.spatial_crop['x1']==0 and 
                video_io.spatial_crop['y1']==0 and
                video_io.spatial_crop['x2']==video_io.video.frame_width and 
                video_io.spatial_crop['y2']==video_io.video.frame_height
            ):
                video_io.get_spatial_crop_input_from_user()
            if video_io.time_crop[0]==0 and video_io.time_crop[1]==video_io.oflow_len:
                video_io.auto_time_crop()

            video_io.draw_diagrams()
            video_io.to_animation()
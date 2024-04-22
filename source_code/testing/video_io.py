
import os, sys

from app.VideoIO import VideoIO

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

VIDEO_NAME = '03-28 used red object/sec' #21 added light and glove/close_startup'

## Test config file handling : make_default_config(), save_config(), load_config()

video_io = VideoIO(data_path, VIDEO_NAME, verbose=10)
print(f'Config at init time: {video_io.config}')

video_io.make_default_config()
print(f'Config after making default config: {video_io.config}')

video_io.save_config()

other_video_io = VideoIO(data_path, VIDEO_NAME)
other_video_io.load_config()
print(f'Config loaded from json file: {other_video_io.config}')

## Test config modifications : auto_time_crop() and get_spatial_crop_input_from_user()

video_io.get_spatial_crop_input_from_user()
print(f'Config after setting spatial crop from user input: {video_io.config}')

video_io.auto_time_crop(verbose=10)
print(f'Config after automatically estimating the time bounds: {video_io.config}')

# Check for correct modification of the json file:
other_video_io.load_config(force=True)
print(f'Config loaded from json file: {other_video_io.config}')

# Test diagram making : draw_diagrams()

video_io.draw_diagrams(show=True, verbose=10)

# Test animation extraction : to_animation()

anim = video_io.to_animation(verbose=10)
anim.display(handles=False, style="lines+markers", doShow=True)

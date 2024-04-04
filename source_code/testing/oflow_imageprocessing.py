

import os, sys

from time import time

t0 = time()

from app.VideoIO import VideoIO, default_config
from app.Color import Color
from app.Video import Video
from app.ImageProcessing import ImageProcessing

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

directory = '03-21 added light and glove' #8 used red object'
assert os.path.exists(data_path+'/'+directory), "Wrong PATH"

video_name = 'far_startup'

complete_dir = dir(ImageProcessing)
methods = []
for attr in complete_dir:
    if '__' not in attr:
        methods.append(attr)
methods.pop(methods.index("none")) if "none" in methods else None
methods.pop(methods.index("rgb")) if "rgb" in methods else None
print(f'Available image processing methods : {methods}')

video_io = VideoIO(f'{data_path}/{directory}/', video_name, verbose=10)

for improcess_method in methods:
    print(f"Current image processing method: {improcess_method}")
    Color.reset()
    video_io.config["image processing method"] = improcess_method
    #print(f"Config : {video_io.config}")
    video_io.process(force=True)
    fig = video_io.draw_diagrams(save=False)
    fig.update_layout(title=improcess_method)
    fig.show()

tf = time()
print(f'Computation took {int(tf-t0)} seconds.') # less than two minutes for 10 videos

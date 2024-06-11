
import os
import numpy as np
import plotly.express as px

from app.video import Video
from app.image_processing import ImageProcessing


data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

directory = '03-28 used red object'
assert os.path.exists(data_path+'/'+directory), "Wrong PATH"

video_name = 'mou'

complete_dir = dir(ImageProcessing)
methods = []
for attr in complete_dir:
    if '__' not in attr:
        methods.append(attr)
print(f'Available image processing methods : {methods}')

image = Video(f'{data_path}/{directory}/{video_name}.mp4', verbose=1).get_frame(0)

fig = px.imshow(image, title="Original image, BGR")
fig.show()

for method in methods:

    res = getattr(ImageProcessing, method)(image)
    color_scale = 'none' if len(res.shape)>2 and res.shape[2]==3 else 'gray'
    fig = px.imshow(res, title=method, color_continuous_scale=color_scale)
    fig.show()


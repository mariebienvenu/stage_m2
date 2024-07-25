
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import  cv2

from app.video_io import VideoIO
from app.image_processing import ImageProcessing
import app.visualisation as vis

def opacity(p):
    return 1-(1-p)**2

PATH = 'C:/Users/Marie Bienvenu/stage_m2/blender_scenes/bouncing_ball/output/'
video_name = "60fps fullHD render"
time_step = 5
threshold = 10

video_io = VideoIO(PATH, video_name, extension=".mp4", verbose=10)
N = video_io.video.frame_count

background = video_io.video.get_frame(N-1, ImageProcessing.rgb).astype(float)
final_image = video_io.video.get_frame(N-1, ImageProcessing.rgb).astype(float)

px.imshow(final_image).show()

for i in range(N//time_step):

    frame_index = time_step*i
    frame = video_io.video.get_frame(frame_index, ImageProcessing.rgb).astype(float)
    cv2.imwrite(f'{PATH}/{frame_index}.jpg',frame)
    diff = np.tile(np.expand_dims(np.sum(np.abs(frame - background), axis=2), axis=2), (1,1,3))
    new = np.where(diff<threshold, final_image, frame)
    p = i/(N//time_step-1)
    final_image = opacity(p)*new + (1-opacity(p))*final_image
    if i<=1:
        fig = px.imshow(frame)
        fig.update_layout(title=f'frame {frame_index}')
        fig.show()
        fig = vis.add_heatmap(diff[:,:,0])
        fig.update_layout(title=f'diff {frame_index}')
        fig.show()
        fig = px.imshow(new)
        fig.update_layout(title=f'new {frame_index}')
        fig.show()
        fig = px.imshow(final_image)
        fig.update_layout(title=f'final {frame_index}')
        fig.show()

fig = px.imshow(final_image)
fig.show()
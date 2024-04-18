import os, sys

def check_sys_path(
        packages_path = "C:/Users/Marie Bienvenu/miniconda3/envs/blender2/Lib/site-packages",
        source_code_path = "C:/Users/Marie Bienvenu/stage_m2/source_code/"
    ):
    if packages_path not in sys.path:
         sys.path.append(packages_path)  # removes package import errors
    if source_code_path not in sys.path:
         sys.path.append(source_code_path)  # removes local import errors

check_sys_path()

import cv2
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from tqdm import tqdm

import bpy

from app.Video import Video
import app.VideoIO as VideoIO
import app.OpticalFlow as OpticalFlow
import app.visualisation as vis
import app.blender_utils as b_utils

import importlib
importlib.reload(b_utils)
importlib.reload(VideoIO)

import app.blender_utils as b_utils
import app.VideoIO as VideoIO

Curve = b_utils.Curve
Animation = b_utils.Animation
m_utils = VideoIO.m_utils

importlib.reload(Curve)

blender_filepath = bpy.data.filepath
scene_path = '/'.join(blender_filepath.split('\\')[:-1])+'/'
assert os.path.exists(scene_path), "Blender scene directory not found."

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

subdirectory = '04-08 appareil de Damien' #'03-11 initial videos'
VIDEO_NAME = 'P1010236'

#video_io = VideoIO.VideoIO(f'{data_path}/{subdirectory}/', VIDEO_NAME, verbose=10)
#video_movement = video_io.to_animation()
video_movement = Animation.Animation().load(f'{data_path}/{subdirectory}/{VIDEO_NAME}/')
frame_times = video_movement[0].get_times()

additionnal_curves_video = Animation.Animation()
for curve in video_movement:
    first_derivative = curve.first_derivative()
    bottom, top = first_derivative.get_value_range()
    if top-bottom > 1e-2:
        additionnal_curves_video.append(first_derivative)

    second_derivative = curve.second_derivative()
    bottom, top = second_derivative.get_value_range()
    if top-bottom > 1e-2:
        additionnal_curves_video.append(second_derivative)
additionnal_curves_video.display(handles=False, style='markers+lines', doShow=True)


NAME = 'Ball'
animation = b_utils.get_animation(NAME)
for curve in animation:
    start, stop = b_utils.get_crop(curve)
    #curve.crop(start, stop)
additionnal_curves = Animation.Animation()

for curve in animation:
    first_derivative = curve.first_derivative(n_samples=frame_times.size-2)
    bottom, top = first_derivative.get_value_range()
    if top-bottom > 1e-2:
        additionnal_curves.append(first_derivative)

    second_derivative = curve.second_derivative(n_samples=frame_times.size-2)
    bottom, top = second_derivative.get_value_range()
    if top-bottom > 1e-2:
        additionnal_curves.append(second_derivative)

#print(additionnal_curves)
resampled_animation = Animation.Animation()
for curve in animation.sample(frame_times.size, start="each", stop="each") + additionnal_curves:
    if len(curve)>1:
        resampled_animation.append(curve)
#print(resampled_animation)

matrix = Animation.Animation.correlate(animation.sample(frame_times.size, start="each", stop="each"), video_movement)

rows = [curve.fullname for curve in animation.sample(frame_times.size, start="each", stop="each")]
columns = [curve.fullname for curve in video_movement]

dataframe = pd.DataFrame(matrix, columns=columns, index=rows)
print(dataframe)


matrix = Animation.Animation.correlate(additionnal_curves, additionnal_curves_video)

rows = [curve.fullname for curve in additionnal_curves]
columns = [curve.fullname for curve in additionnal_curves_video]

dataframe = pd.DataFrame(matrix, columns=columns, index=rows)
print(dataframe)

resampled_animation.display(style='markers+lines', handles=False, doShow=True)
video_movement.display(style='markers+lines', handles=False, doShow=True)

# TODO: make a figure with the two animations on different subplots ; make a vis.py function to handle layout ? (name of axis, etc)

from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=['Video curves', 'Blender curves'])
video_movement.display(style='markers+lines', row=1, col=1, handles=False, fig=fig)
resampled_animation.display(style='markers+lines', row=2, col=1, handles=False, fig=fig, doShow=True)


## Display rescaled translation y (video) on same plot than translation z (blender)

transl_blender = resampled_animation.find("location Z sampled")
fig = transl_blender.display(handles=False, style='markers+lines') # is in frame, 24fps

transl_video = video_movement.find("Location Y")

mean_diff = np.mean(transl_blender.get_values()) - np.mean(transl_video.get_values())
std_diff = np.std(transl_blender.get_values()) / np.std(transl_video.get_values())

def affine_transform(curve : Curve.Curve):
    curve.value_transl(mean_diff)
    curve.value_scale(center=np.mean(curve.get_values()), scale=std_diff)

affine_transform(transl_video)

transl_video.time_transl(1-np.min(transl_video.get_times())) # put start at frame 1

transl_video.display(handles=False, style='markers+lines', fig=fig, doShow=True)

print(f'Correlation between retimed/rescaled curves : {m_utils.correlation(transl_video.get_values(), transl_blender.get_values())}') # is same as element of matrix because correlation coefficient is scale & translation independent



# Visualisation of oflow magnitude&angle and blender anim translation Z on top of each other: 

magnitude = video_movement.find('Oflow magnitude - mean')
angle = video_movement.find('Oflow angle - mean')
transl_z = animation.find('location Z')
sampled_transl_z = resampled_animation.find('location Z sampled')

# on remet le début à frame 1
magnitude.time_transl(1-np.min(magnitude.get_times()))
angle.time_transl(1-np.min(angle.get_times()))

anim_start = np.min(transl_z.get_times())
transl_z.time_transl(1-anim_start)
sampled_transl_z.time_transl(1-anim_start)

# affichage
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=["Angle of video's optical flow", "Magnitude of video's optical flow", "Blender animations's translation curve"])
angle.display(fig=fig, handles=False, style="lines + markers", row=1, col=1)
magnitude.display(fig=fig, handles=False, style="lines + markers", row=2, col=1)
transl_z.display(fig=fig, handles=True, row=3, col=1)
sampled_transl_z.display(fig=fig, handles=False, style='lines', row=3, col=1)
fig.update_yaxes(title_text="angle (degree)", row=1, col=1)
fig.update_yaxes(title_text="magnitude (pixel)", row=2, col=1)
fig.update_yaxes(title_text="magnitude (blender unit)", row=3, col=1)
fig.update_xaxes(title_text="time (frame)", row=3, col=1)

fig.write_html(f'{data_path}/{subdirectory}/{VIDEO_NAME}_comparison.html')

fig.show()

# Visualisation of velocity y and blender anim derivative of translation Z on top of each other: 

velo_y = video_movement.find('Velocity Y')
transl_z_prime = additionnal_curves.find('First derivative of location Z')

# on remet le début à frame 1
velo_y.time_transl(1-np.min(velo_y.get_times()))

anim_start = np.min(transl_z_prime.get_times())
transl_z_prime.time_transl(1-anim_start)

# affichage
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Y Velocity of video", "First derivative of Blender animation's translation curve"])
velo_y.display(fig=fig, handles=False, style="lines + markers", row=1, col=1)
transl_z_prime.display(fig=fig, handles=False, style="lines + markers", row=2, col=1)
fig.update_yaxes(title_text="magnitude (pixels/second)", row=1, col=1)
fig.update_yaxes(title_text="magnitude (blender unit)", row=2, col=1)
fig.update_xaxes(title_text="time (frame)", row=2, col=1)

fig.write_html(f'{data_path}/{subdirectory}/{VIDEO_NAME}_comparison_velocity.html')

fig.show()

# Viualisation of Acceleration y and blender anim second derivative of translation Z on top of each other
# with blender translation below

acc_y = additionnal_curves_video.find('First derivative of Velocity Y')
transl_z_seconde = additionnal_curves.find('Second derivative of location Z')

# on remet le début à frame 1
acc_y.time_transl(1-np.min(acc_y.get_times()))

anim_start = np.min(transl_z_seconde.get_times())
transl_z_seconde.time_transl(1-anim_start)

fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Blender animation's translation curve", "Accelerations"])

transl_z.display(fig=fig2, handles=True, style="markers", row=1, col=1)
sampled_transl_z.display(fig=fig2, handles=False, style='lines', row=1, col=1)

acc_y.display(fig=fig2, handles=False, style='lines+markers', row=2, col=1)
transl_z_seconde.display(fig=fig2, handles=False, style='lines+markers', row=2, col=1)

fig2.update_yaxes(title_text="magnitude (blender unit)", row=1, col=1)
fig2.update_yaxes(title_text="magnitude (/second²)", row=2, col=1)
fig2.update_xaxes(title_text="time (frame)", row=2, col=1)

fig2.write_html(f'{data_path}/{subdirectory}/{VIDEO_NAME}_comparison_acceleration.html')

fig2.show()
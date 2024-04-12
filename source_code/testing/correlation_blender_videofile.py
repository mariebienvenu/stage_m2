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
import app.optical_flow as optical_flow
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
    sampling_step = (curve.time_range[1]-curve.time_range[0])/frame_times.size
    if sampling_step == 0:
        continue
    
    first_derivative = m_utils.derivee(curve.get_values(), sampling_step)
    if np.max(first_derivative)-np.min(first_derivative)  > 1e-2 : # variation in derivative
        coordinates = np.vstack((curve.get_times()[1:], first_derivative)).T
        additionnal_curves_video.append(Curve.Curve(coordinates, fullname=f'First derivative of {curve.fullname}'))

additionnal_curves_video.display(handles=False, style='markers+lines', doShow=True)


NAME = 'Ball'
animation = b_utils.get_animation(NAME)
for curve in animation:
    start, stop = b_utils.get_crop(curve)
    #curve.crop(start, stop)
additionnal_curves = Animation.Animation()

for curve in animation:
    sampling_step = (curve.time_range[1]-curve.time_range[0])/(frame_times.size-1)
    if sampling_step == 0:
        continue
    delta_t = sampling_step/20
    fcurve : bpy.types.FCurve = curve.pointer
    sampling_t = [curve.time_range[0] + i*sampling_step for i in range(frame_times.size)]
    sampling_v = [fcurve.evaluate(t) if i%2==0 else fcurve.evaluate(t+delta_t)  for (i,t) in enumerate([sampling_t[j//2] for j in range(2*(frame_times.size))])]

    first_derivative = m_utils.derivee(sampling_v, delta_t)[::2]
    absolute_first_derivative = np.abs(first_derivative)
    if np.max(absolute_first_derivative)-np.min(absolute_first_derivative)  > 1e-2 : # variation in derivative
        coordinates = np.vstack((sampling_t, absolute_first_derivative)).T
        additionnal_curves.append(Curve.Curve(coordinates, fullname=f'absolute first derivative of {curve.fullname}'))

        coordinates = np.vstack((sampling_t, first_derivative)).T
        additionnal_curves.append(Curve.Curve(coordinates, fullname=f'first derivative of {curve.fullname}'))

#print(additionnal_curves)
resampled_animation = Animation.Animation()
for curve in animation.sample(frame_times.size, start="each", stop="each") + additionnal_curves:
    if len(curve)>1:
        resampled_animation.append(curve)
#print(resampled_animation)

def compare_animations(anim1:Animation.Animation, anim2:Animation.Animation):
    correlation_matrix = np.zeros((len(anim1), len(anim2)), dtype=np.float64)
    for i, curve1 in enumerate(anim1):
        for j, curve2 in enumerate(anim2):
            assert len(curve1) == len(curve2), f"Cannot compare animation curves of different length: {len(curve1)} != {len(curve2)}"
            values1 = curve1.get_values()
            values2 = curve2.get_values()
            correlation_matrix[i,j] = m_utils.correlation(values1, values2)
    return correlation_matrix

matrix = compare_animations(resampled_animation, video_movement)

rows = [curve.fullname for curve in resampled_animation]
columns = [curve.fullname for curve in video_movement]

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
transl_z_prime = additionnal_curves.find('first derivative of location Z')

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
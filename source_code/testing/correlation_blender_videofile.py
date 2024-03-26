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
import app.optical_flow as optical_flow
import app.visualisation as vis
import app.blender_utils as b_utils
import app.maths_utils as m_utils

import importlib
importlib.reload(b_utils)
importlib.reload(m_utils)

import app.blender_utils as b_utils
import app.maths_utils as m_utils

Curve = b_utils.Curve
Animation = b_utils.Animation

blender_filepath = bpy.data.filepath
scene_path = '/'.join(blender_filepath.split('\\')[:-1])+'/'
assert os.path.exists(scene_path), "Blender scene directory not found."

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

VIDEO_NAME = '03-11 initial videos/souris.mp4' #'03-21 added light and glove/close_startup.mp4'
video = Video(data_path + f'{VIDEO_NAME}', verbose=1)

oflow_len = video.frame_count - 1
frame_times = np.array(list(range(oflow_len)))

oflows = np.zeros((oflow_len, video.frame_height, video.frame_width, 2), dtype=np.float64)
magnitudes = np.zeros((oflow_len, video.frame_height, video.frame_width), dtype=np.float64)
angles = np.zeros((oflow_len, video.frame_height, video.frame_width), dtype=np.float64)

for index in tqdm(range(oflow_len), desc='Oflow computation'):
    frame1 = cv2.cvtColor(video.get_frame(index), cv2.COLOR_BGR2GRAY)
    frame2 = cv2.cvtColor(video.get_frame(index+1), cv2.COLOR_BGR2GRAY)

    oflow = optical_flow.compute_oflow(frame1, frame2, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2)
    oflows[index,:,:,:] = oflow

    magnitude, angle = optical_flow.cartesian_to_polar(oflow, degrees=True)
    angles[angles>180] -= 360
    magnitudes[index,:,:] = magnitude
    angles[index,:,:] = angle

magnitude_means = np.zeros(oflow_len, dtype=np.float64)
magnitude_stds = np.zeros(oflow_len, dtype=np.float64)

angle_means = np.zeros(oflow_len, dtype=np.float64)
angle_stds = np.zeros(oflow_len, dtype=np.float64)

for index in range(oflow_len):
    mag, ang = magnitudes[index,...], angles[index,...]
    h = optical_flow._get_threshold(mag, 0.95)
    mag_filtered, ang_filtered = mag[mag>h], ang[mag>h]
    mag_values, ang_values = np.ravel(mag_filtered), np.ravel(ang_filtered)

    measures = optical_flow.measure_oflow(mag_values, ang_values)

    magnitude_means[index] = measures['magnitude_mean']
    magnitude_stds[index] = measures['magnitude_std']

    angle_means[index] = measures['angle_mean']
    angle_stds[index] = measures['angle_std']

start, stop = optical_flow.get_crop(frame_times, magnitude_means, patience=2)

position_y = m_utils.integrale3(magnitude_means*np.sin(-angle_means*np.pi/180), step=1) # reverse angles because up is - in image space
position_x = m_utils.integrale3(magnitude_means*np.cos(-angle_means*np.pi/180), step=1) # reverse angles because up is - in image space

video_movement = Animation([
    Curve(np.vstack((frame_times, position_x)).T, fullname='gesture position x'),
    Curve(np.vstack((frame_times, position_y)).T, fullname='gesture position y'),
    Curve(np.vstack((frame_times, np.abs(magnitude_means*np.sin(-angle_means*np.pi/180)))).T, fullname='gesture speed x'),
    Curve(np.vstack((frame_times, np.abs(magnitude_means*np.cos(-angle_means*np.pi/180)))).T, fullname='gesture speed y')
])

video_movement.crop(start, stop )

frame_times = video_movement[0].get_times()

NAME = 'Ball'
animation = b_utils.get_animation(NAME)
for curve in animation:
    start, stop = b_utils.get_crop(curve)
    curve.crop(start, stop)
additionnal_curves = Animation()

for curve in animation:
    sampling_step = (curve.time_range[1]-curve.time_range[0])/(frame_times.size+1)
    fcurve = curve.pointer
    sampling_t = [curve.time_range[0] + i*sampling_step for i in range(frame_times.size+1)]
    sampling_v = [fcurve.evaluate(t) for t in sampling_t]

    absolute_first_derivative = np.abs(m_utils.derivee(sampling_v, sampling_step))
    if np.max(absolute_first_derivative)-np.min(absolute_first_derivative)  > 1e-2 : # variation in derivative
        coordinates = np.vstack((sampling_t[1:], absolute_first_derivative)).T
        additionnal_curves.append(Curve(coordinates, fullname=f'absolute first derivative of {curve.fullname}'))

#print(additionnal_curves)
resampled_animation = animation.resample(frame_times.size) + additionnal_curves
#print(resampled_animation)

def compare_animations(anim1:Animation, anim2:Animation):
    correlation_matrix = np.zeros((len(anim1), len(anim2)), dtype=np.float64)
    for i, curve1 in enumerate(anim1):
        for j, curve2 in enumerate(anim2):
            assert len(curve1) == len(curve2), f"Cannot compare animation curves of different length: {len(curve1)} != {len(curve2)}"
            values1 = curve1.get_values()
            values2 = curve2.get_values()
            correlation_matrix[i,j] = m_utils.correlation(values1, values2)
    return correlation_matrix

matrix = compare_animations(resampled_animation, video_movement)
#print(matrix)

rows = [curve.fullname for curve in resampled_animation]
columns = [curve.fullname for curve in video_movement]

dataframe = pd.DataFrame(matrix, columns=columns, index=rows)
print(dataframe)

resampled_animation.display(style='markers+lines', handles=False, doShow=True)
video_movement.display(style='markers+lines', handles=False, doShow=True)
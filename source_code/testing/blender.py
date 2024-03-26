import sys
import os

def check_sys_path(
        packages_path = "C:/Users/Marie Bienvenu/miniconda3/envs/blender2/Lib/site-packages",
        source_code_path = "C:/Users/Marie Bienvenu/stage_m2/source_code/"
    ):
    
    if packages_path not in sys.path:
         sys.path.append(packages_path)  # removes package import errors
    if source_code_path not in sys.path:
         sys.path.append(source_code_path)  # removes local import errors

check_sys_path()

import bpy

import cv2 
import numpy as np
import plotly.graph_objects as go

import app.blender_utils as b_utils
import app.maths_utils as m_utils

import importlib
importlib.reload(b_utils)

blender_filepath = bpy.data.filepath
scene_path = '/'.join(blender_filepath.split('\\')[:-1])+'/'
assert os.path.exists(scene_path), "Blender scene directory not found."

## Test of b_utils.get_animation()

NAME = 'Ball'
animation = b_utils.get_animation(NAME)
print(animation)
print(b_utils.get_animation("not in scene"))

fig = go.Figure()
animation.display(fig=fig)

for curve in animation:
    sampling_step = 0.5
    sampling_t = [curve.time_range[0] + i*sampling_step for i in range(int((curve.time_range[1]-curve.time_range[0])/sampling_step))]
    sampling_v = [curve.pointer.evaluate(t) for t in sampling_t]
    fig.add_trace(go.Scatter(
        x = sampling_t,
        y = sampling_v,
        name = f'interpolation of {curve.fullname}',
        mode="lines",
        marker_color = f'rgb{curve.color}'
    ))

    first_derivative = m_utils.derivee(sampling_v, sampling_step)
    if np.max(first_derivative)-np.min(first_derivative)  > 1e-2 : # variation in derivative
        fig.add_trace(go.Scatter(
            x = sampling_t[1:],
            y = np.abs(first_derivative),
            name = f'absolute first derivative of {curve.fullname}',
            mode="lines"
        ))

fig.write_html(scene_path+f'{NAME}_animation.html')
fig.show()

## Test of b_utils.get_crop()

y = animation[1]
start, stop = b_utils.get_crop(y)

fig2 = y.display()

sampling_step = 0.5
sampling_t = [y.time_range[0] + i*sampling_step for i in range(int((y.time_range[1]-y.time_range[0])/sampling_step))]
sampling_v = [y.pointer.evaluate(t) for t in sampling_t]
fig2.add_trace(go.Scatter(
        x = sampling_t,
        y = sampling_v,
        name = f'interpolation of {y.fullname}',
        mode="lines",
        marker_color = f'rgb{y.color}'
    ))

fig2.add_vline(x=start, annotation_text="Start", annotation_position="top right")
fig2.add_vline(x=stop, annotation_text="Stop", annotation_position="top right")

fig2.write_html(scene_path+f'{NAME}_height_crop.html')
fig2.show()

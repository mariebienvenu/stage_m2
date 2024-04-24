
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

from sklearn.cross_decomposition import CCA
import numpy as np
import plotly.graph_objects as go
import pandas as pd

import bpy

import app.VideoIO as VideoIO
import app.blender_utils as b_utils

import importlib
importlib.reload(b_utils)
importlib.reload(b_utils.Curve)
importlib.reload(VideoIO)

import app.blender_utils as b_utils
import app.VideoIO as VideoIO

Curve = b_utils.Curve
Animation = b_utils.Animation
m_utils = VideoIO.m_utils
vis = VideoIO.vis


blender_filepath = bpy.data.filepath
scene_path = '/'.join(blender_filepath.split('\\')[:-1])+'/'
assert os.path.exists(scene_path), "Blender scene directory not found."

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

subdirectory = '04-08 appareil de Damien' #'03-11 initial videos'
VIDEO_NAME = 'P1010236'

video_movement = Animation.Animation().load(f'{data_path}/{subdirectory}/{VIDEO_NAME}/')

frame_times = video_movement[0].get_times()

NAME = 'Ball'
animation = b_utils.get_animation(NAME)
for curve in animation:
    start, stop = curve.get_auto_crop()
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

resampled_animation = Animation.Animation()
for curve in animation.sample(frame_times.size, start="each", stop="each") + additionnal_curves:
    if len(curve)>1:
        resampled_animation.append(curve)

target_curve = resampled_animation.find('first derivative of location Z')

user_features = np.array([curve.get_values() for curve in video_movement]).T
target = np.expand_dims(target_curve.get_values(), axis=0).T

cca = CCA(n_components=1)
cca.fit(user_features, target)

transformed_features = cca.predict(user_features)


vis.Color.reset()
vis.Color.next()

fig = go.Figure()
fig.add_trace(go.Scatter(y=np.ravel(transformed_features), x=target_curve.get_times(), mode="markers+lines"))
target_curve.display(handles=False, style='markers+lines', fig=fig)
fig.show()

video_movement.display(handles=False, style='lines+markers', doShow=True)
resampled_animation.display(handles=False, style='lines+markers', doShow=True)

def get_cca_df(cca : CCA, feature_names):
    matrix = np.squeeze(np.array([cca.coef_.T, cca.x_weights_, cca.x_loadings_, cca.x_rotations_]), axis=2).T
    columns = ["coefficients", "X weights", "X loadings", "X rotations"]
    return pd.DataFrame(matrix, columns=columns, index=feature_names)

rows = [curve.fullname for curve in video_movement]
print(get_cca_df(cca, rows))
print(f"SCORE : {cca.score(user_features, target)}")


## Now with only the Y translation

user_curve = video_movement.find('Velocity Y')

user_features = np.expand_dims(user_curve.get_values(), axis=1)
target = np.expand_dims(target_curve.get_values(), axis=1)

other_cca = CCA(n_components=1)
other_cca.fit(user_features, target)

transformed_features = other_cca.predict(user_features)


vis.Color.reset()
vis.Color.next()

fig = go.Figure()
fig.add_trace(go.Scatter(y=np.ravel(transformed_features), x=target_curve.get_times(), mode="markers+lines"))
target_curve.display(handles=False, style='markers+lines', fig=fig)
fig.show()

rows = [user_curve.fullname]
print(get_cca_df(other_cca, rows))
print(f"SCORE : {other_cca.score(user_features, target)}")
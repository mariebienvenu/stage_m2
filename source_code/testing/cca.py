
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

subdirectory = '03-11 initial videos'
VIDEO_NAME = 'souris'

video_movement = Animation.Animation().load(f'{data_path}/{subdirectory}/{VIDEO_NAME}/')

frame_times = video_movement[0].get_times()

NAME = 'Ball'
animation = b_utils.get_animation(NAME)
for curve in animation:
    start, stop = b_utils.get_crop(curve)
    curve.crop(start, stop)

resampled_animation = animation.sample(frame_times.size, start="each", stop="each")

target_curve = resampled_animation.find('location Z sampled')

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

user_curve = video_movement.find('Location Y')

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
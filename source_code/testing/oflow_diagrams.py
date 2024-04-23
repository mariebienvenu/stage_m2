import os

import cv2
import numpy as np

from tqdm import tqdm
from plotly.subplots import make_subplots

from app.Video import Video
import app.OpticalFlow as OpticalFlow
import app.maths_utils as m_utils
import app.visualisation as vis

from app.Animation import Animation
from app.Curve import Curve

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

VIDEO_NAME = '03-28 used red object/mou' #11 initial videos/souris' #21 added light and glove/far_startdown'
video = Video(data_path + VIDEO_NAME +'.mp4', verbose=1)
#video.play_frame_by_frame()

oflow_len = video.frame_count - 1
frame_times = np.arange(0, oflow_len/video.fps, 1/video.fps)

flows = [video.get_optical_flow(index) for index in tqdm(range(oflow_len), desc='Oflow computation')]

magnitude_means = np.array([flow.get_measure(OpticalFlow.Measure.MAGNITUDE_MEAN) for flow in flows])
magnitude_stds = np.array([flow.get_measure(OpticalFlow.Measure.MAGNITUDE_STD) for flow in flows])
angle_means = np.array([flow.get_measure(OpticalFlow.Measure.ANGLE_MEAN) for flow in flows])
angle_stds = np.array([flow.get_measure(OpticalFlow.Measure.ANGLE_STD) for flow in flows])

# Calcul de l'intégrale du flux i.e. la position

velocity_x, velocity_y = OpticalFlow.polar_to_cartesian(magnitude_means, -angle_means, degrees=True) # reverse angles because up is - in image space
velocity_x, velocity_y = np.ravel(velocity_x), np.ravel(velocity_y)
position_x, position_y = m_utils.integrale3(velocity_x, step=1), m_utils.integrale3(velocity_y, step=1)

## Création et enregistrement de l'animation avec toutes les courbes

anim = Animation([
    Curve(np.vstack((frame_times, magnitude_means)).T, fullname='Oflow magnitude - mean'),
    Curve(np.vstack((frame_times, magnitude_stds)).T, fullname='Oflow magnitude - std'),
    Curve(np.vstack((frame_times, angle_means)).T, fullname='Oflow angle - mean'),
    Curve(np.vstack((frame_times, angle_stds)).T, fullname='Oflow angle - std'),
    Curve(np.vstack((frame_times, velocity_x)).T, fullname='Velocity X'),
    Curve(np.vstack((frame_times, velocity_y)).T, fullname='Velocity Y'),
    Curve(np.vstack((frame_times, position_x)).T, fullname='Location X'),
    Curve(np.vstack((frame_times, position_y)).T, fullname='Location Y'),
])
anim.save(data_path + VIDEO_NAME + '/')

## Estimation des bornes

start, stop = Animation.find('Oflow magnitude - mean').get_auto_crop(use_handles=False)
start2, stop2 = Animation.find('Oflow magnitude - mean').get_auto_crop(use_handles=False, patience=2)

## Visualisation globale

fig = make_subplots(rows=2, cols=3, subplot_titles=("Amplitude du flux au cours du temps", "Vitesses au cours du temps" , "Portraits de phase", "Angle du flux au cours du temps", "Positions au cours du temps", "Trajectoire"))

vis.magnitude_angle(frame_times, magnitude_means, magnitude_stds, angle_means, angle_stds, fig=fig, rows=[1,2], cols=[1,1])
vis.add_curve(y=velocity_y, x=position_y, name="y'=f(y) - Portrait de phase de Y", fig=fig, col=3, row=1)
vis.add_curve(y=velocity_x, x=position_x, name="x'=f(x) - Portrait de phase de X", fig=fig, col=3, row=1)
vis.add_curve(velocity_y, x=frame_times, name="y=f(t) - Velocity along X axis", fig=fig, col=2, row=1)
vis.add_curve(velocity_x, x=frame_times, name="x=f(t) - Velocity along X axis", fig=fig, col=2, row=1)
vis.add_curve(position_y, x=frame_times, name="y=f(t) - Translation along X axis", fig=fig, col=2, row=2)
vis.add_curve(position_x, x=frame_times, name="x=f(t) - Translation along X axis", fig=fig, col=2, row=2)
vis.add_curve(y=position_y, x=position_x, name="y=f(x) - Trajectoire", fig=fig, col=3, row=2)

rows, cols = (1,1,2,2), (1,2,1,2)
for row, col in zip(rows, cols):
    fig.add_vline(x=start, row=row, col=col)
    fig.add_vline(x=stop, row=row, col=col)

if start2 != start:
    for row, col in zip(rows, cols):
        fig.add_vline(x=start2, line_dash="dash", row=row, col=col)
if stop2 != stop:
    for row, col in zip(rows, cols):
        fig.add_vline(x=stop2, line_dash="dash", row=row, col=col)

fig.update_layout(title=f'Optical flow  - {VIDEO_NAME}')
fig.write_html(data_path+f"/{VIDEO_NAME}_diagram.html")
fig.show()

## Visualisation du flux

fig1 = vis.magnitude_angle(frame_times, magnitude_means, magnitude_stds, angle_means, angle_stds)

fig1.add_vline(x=start, annotation_text="Start, no patience", annotation_position="top right")
fig1.add_vline(x=stop, annotation_text="Stop, no patience", annotation_position="top right")

if start2 != start:
    fig1.add_vline(x=start2, line_dash="dash", annotation_text="Start, patience=2", annotation_position="top right")
if stop2 != stop:
    fig1.add_vline(x=stop2, line_dash="dash", annotation_text="Stop, patience=2", annotation_position="top right")

fig1.update_layout(title=f'Optical flow  - {VIDEO_NAME}')
fig1.write_html(data_path+f"/{VIDEO_NAME}_oflow.html")
fig1.show()
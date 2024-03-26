import os

import cv2
import numpy as np

from tqdm import tqdm

from app.Video import Video
import app.optical_flow as optical_flow
import app.maths_utils as m_utils
import app.visualisation as vis

import plotly.graph_objects as go

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

VIDEO_NAME = '03-11 initial videos/souris' #21 added light and glove/far_startdown'
video = Video(data_path + VIDEO_NAME +'.mp4', verbose=1)
#video.play_frame_by_frame()

oflow_len = video.frame_count - 1

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

    magnitude_means[index] = np.mean(mag_values)
    magnitude_stds[index] = np.std(mag_values)

    angle_means[index] = np.mean(mag_values*ang_values)/magnitude_means[index]
    angle_stds[index] = np.std(ang_values) # following formula gives way too big values : np.std(mag_values*ang_values)/magnitude_stds[index]

    '''
    plt.figure()
    plt.hist(x=values, bins=200)
    plt.xlim([0, 60])
    plt.ylim([0, 2000])
    plt.savefig(data_path+f'/magnitude_histograms/frame_{index}.png')
    plt.close()
    '''

start, stop = optical_flow.get_crop(magnitude_means)
start2, stop2 = optical_flow.get_crop(magnitude_means, patience=2)

fig = vis.magnitude_angle(
    magnitude_means,    
    magnitude_stds,
    angle_means,
    angle_stds,
    oflow_len,
    VIDEO_NAME,
    data_path
)

fig.add_vline(x=start, annotation_text="Start, no patience", annotation_position="top right")
fig.add_vline(x=stop, annotation_text="Stop, no patience", annotation_position="top right")

if start2 != start:
    fig.add_vline(x=start2, line_dash="dash", annotation_text="Start, patience=2", annotation_position="top right")
if stop2 != stop:
    fig.add_vline(x=stop2, line_dash="dash", annotation_text="Stop, patience=2", annotation_position="top right")

fig.write_html(data_path+f"/{VIDEO_NAME}_oflow.html")
#fig.show()

## Intégrale première : trajectoire

position_y = m_utils.integrale3(magnitude_means*np.sin(-angle_means*np.pi/180), step=1) # reverse angles because up is - in image space
position_x = m_utils.integrale3(magnitude_means*np.cos(-angle_means*np.pi/180), step=1) # reverse angles because up is - in image space


fig2 = vis.add_curve(position_y, color='rgb(100,100,0)', name="Translation along Y axis")
vis.add_curve(position_x, color='rgb(255,0,255)', name="Translation along X axis", fig=fig2)
fig2.write_html(data_path+f"/{VIDEO_NAME}_trajectory.html")
#fig2.show() 

# TODO Quid d'un portrait de phase ?
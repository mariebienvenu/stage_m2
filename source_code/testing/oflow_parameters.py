import os

import cv2

import plotly.express as px
import plotly.graph_objects as go

from app.Video import Video
import app.OpticalFlow as OpticalFlow

data_path = 'C:/Users/Marie Bienvenu/Dropbox/#TRANSMISSION/stage 4a/'
assert os.path.exists(data_path), "Wrong PATH"

video = Video(data_path + 'souris.mp4', verbose=1)
video.play_frame_by_frame()

index = 58
frame1 = cv2.cvtColor(video.get_frame(index), cv2.COLOR_BGR2GRAY)
frame2 = cv2.cvtColor(video.get_frame(index+1), cv2.COLOR_BGR2GRAY)

cv2.imshow("gray", frame1)
cv2.waitKey(0)

fig = go.Figure(data=[go.Surface(z=(frame1-frame2)**2)])
fig.show()

for level in range(1,101):
    oflow = OpticalFlow.compute_oflow(frame1, frame2, levels=level)
    bgr = OpticalFlow.make_oflow_image(oflow)
    cv2.imwrite(data_path+f'/optical_flows_levels/level_{level}.png', bgr)

for winsize in range(15,700, 10):
    oflow = OpticalFlow.compute_oflow(frame1, frame2, winsize=winsize)
    bgr = OpticalFlow.make_oflow_image(oflow)
    cv2.imwrite(data_path+f'/optical_flows_window_size/winsize_{winsize}.png', bgr)

for iterations in range(1,32):
    oflow = OpticalFlow.compute_oflow(frame1, frame2, iterations=iterations)
    bgr = OpticalFlow.make_oflow_image(oflow)
    cv2.imwrite(data_path+f'/optical_flows_iterations/iterations_{iterations}.png', bgr)

for polynomial in range(3,100, 2):
    oflow = OpticalFlow.compute_oflow(frame1, frame2, poly_n=polynomial, poly_sigma=1.3+(polynomial-2)*0.25)
    bgr = OpticalFlow.make_oflow_image(oflow)
    cv2.imwrite(data_path+f'/optical_flows_polynomials/poly_n_{polynomial}.png', bgr)

#oflow = compute_oflow(frame1, frame2, poly_n=97, poly_sigma=1.3+(97-2)*0.25, winsize=200, levels=64, iterations=32)

import plotly.graph_objects as go
import plotly.figure_factory as ff

import numpy as np

from app.video_io import VideoIO
import app.visualisation as vis
from app.optical_flow import Measure
import app.maths_utils as m_utils

PATH = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/06-11 claquements en nombres/'

filename = "P1010261" # ou P1010263

video = VideoIO(PATH, filename, verbose=3)

flow = video.get_oflow(143)

print(np.max(flow.magnitude))
print(flow.get_measure(Measure.MAGNITUDE_MEAN))

vis.add_heatmap(flow.magnitude[::-1,:], doShow=True)


flow = video.get_oflow(143+125)

print(np.max(flow.magnitude))
print(flow.get_measure(Measure.MAGNITUDE_MEAN))

vis.add_heatmap(flow.magnitude[::-1,:], doShow=True)

video.process()

curve = video.data['Oflow magnitude - mean 1']
fig0 = vis.add_curve(curve)
fig0.show()

curve2 = m_utils.derivee_seconde(curve, step=1)
fig1 = vis.add_curve(curve2)
fig1.show()


'''

u,v = flow.x, flow.y
n,m = u.shape

x,y = np.meshgrid(np.arange(0, m, 1), np.arange(0, n, 1))

fig = ff.create_quiver(x[210:240, 210:240], y[210:240, 210:240], u[210:240, 210:240], v[210:240, 210:240])
fig.show()

print(2)'''
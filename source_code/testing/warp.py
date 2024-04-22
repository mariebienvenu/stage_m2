
import numpy as np
import plotly.graph_objects as go
import cv2

import app.Warp as Warp

DO_SHOW = True

## Test de LinearWarp1D

X_in = np.arange(-10, 10, 0.5)
X_out = np.cos(X_in/5)

Warp1D = Warp.LinearWarp1D(X_in, X_out)

inputs = [-20, 30, 0.3, 7.32]
outputs = Warp1D(inputs)

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_in, y=X_out, mode='markers+lines'))
fig.add_trace(go.Scatter(x=inputs, y=outputs, mode='markers'))
if DO_SHOW : fig.show()

## Test de LinearWarp2D

grid = np.array([[e for _ in range(40)] for e in np.arange(-10, 10, 0.5)])
X_in = np.ravel(grid)
Y_in = np.ravel(grid.T)
X_out = np.cos(X_in/5)
Y_out = np.sin(Y_in/5)

Warp2D = Warp.LinearWarp2D(X_in, Y_in, X_out, Y_out)

x_inputs = [-6.3, -2.7, 0.3, 7.32]
y_inputs = [-9.7, 0.3, 6.6, -2.7]
x_outputs, y_outputs = Warp2D(x_inputs, y_inputs)

def get_colors(x,y):
    L = int(np.sqrt(x.size))
    h = (x+1)*180
    s = np.ones_like(h)*255
    v = (y+1)/2*256
    hsv = np.vstack((h,s,v)).T
    grid = cv2.cvtColor(hsv.reshape(L,L, 3).astype(np.uint8), cv2.COLOR_HSV2RGB).reshape(L**2, 3)
    return [f'rgb{tuple(grid[i,:])}' for i in range(L**2)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_in, y=Y_in, mode="markers", marker_color=get_colors(X_out, Y_out), marker_size=15))
fig.add_trace(go.Scatter(x=x_inputs, y=y_inputs, mode="lines+markers", marker_color=get_colors(x_outputs, y_outputs), marker_size=40))
if DO_SHOW : fig.show()
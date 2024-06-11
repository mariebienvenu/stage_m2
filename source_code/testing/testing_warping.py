
import numpy as np
import plotly.graph_objects as go
import cv2

import app.warping as W
from app.color import Color

DO_SHOW = True

## Test des Warp1D

X_in = np.linspace(-10, 10, 11)
X_out = np.cos(X_in/5)

cubic_warp = W.CubicWarp1D(X_in, X_out)
linear_warp = W.LinearWarp1D(X_in, X_out)

inputs = np.linspace(-15, 15, 151)
cubic_outputs,_ = cubic_warp(inputs, None)
linear_outputs,_ = linear_warp(inputs, None)

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_in, y=X_out, mode='markers', name="control points"))
fig.add_trace(go.Scatter(x=inputs, y=cubic_outputs, mode='lines', name="cubic interpolation"))
fig.add_trace(go.Scatter(x=inputs, y=linear_outputs, mode='lines', name="linear interpolation"))
if DO_SHOW : fig.show()

## Test de LinearWarp2D

grid = np.array([[e for _ in range(40)] for e in np.arange(-10, 10, 0.5)])
X_in = np.ravel(grid)
Y_in = np.ravel(grid.T)
X_out = np.cos(X_in/5)
Y_out = np.sin(Y_in/5)

Warp2D = W.LinearWarp2D(X_in, Y_in, X_out, Y_out)
print(Warp2D)

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
    return [Color.to_string(tuple(grid[i,:])) for i in range(L**2)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=X_in, y=Y_in, mode="markers", marker_color=get_colors(X_out, Y_out), marker_size=15))
fig.add_trace(go.Scatter(x=x_inputs, y=y_inputs, mode="lines+markers", marker_color=get_colors(x_outputs, y_outputs), marker_size=40))
if DO_SHOW : fig.show()

## Test du getter

X_in = np.linspace(-10, 10, 11)
X_out = np.cos(X_in/5)

warp1 = W.make_warp(dimension=1, interpolation='linear', X_in=X_in, X_out=X_out)
warp2 = W.make_warp(dimension=1, interpolation='cubic', X_in=X_in, X_out=X_out)
print(warp1, warp2)
try:
    warp3 = W.make_warp(dimension=1, interpolation='stuff', X_in=X_in, X_out=X_out)
except NotImplementedError as e:
    print(f"Raised: {e} (as expected)")

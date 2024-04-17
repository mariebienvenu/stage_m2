
import numpy as np
from random import random
import pandas as pd
import plotly.graph_objects as go

DO_SHOW=False

## Test of correlation

from app.maths_utils import correlation

arr1 = np.array([1,2,3])
arr2 = np.array([2,4,6])
corr = correlation(arr1, arr2)
print(corr)

arr1 = np.arange(0, 10, 0.1) # size 100
arr2 = arr1*(-0.5)
arr3 = np.exp(arr1)
arr4 = np.random.random(arr1.size) # random

print(correlation(arr1, arr2), correlation(arr1, arr3), correlation(arr1, arr4)) # expect -1, something, 0

arr5 = random()*arr1 + random()
print(correlation(arr1, arr5)) # expect 1

## Test of Dynamic Time Warping

from app.maths_utils import dynamic_time_warping
import app.visualisation as vis

# SIMPLE CASE
y1 = np.array([1, 2, 3, 2, 3, 2, 1, 1])
y2 = np.array([1, 1, 2, 3, 2, 3, 2, 1])
score, pairings, DTW  = dynamic_time_warping(y1, y2, debug=True)
print(f"DTW score in ideal case: {score:.2f}")
print(f"Pairings found: {pairings}")
print(f"Computed table: \n{pd.DataFrame(DTW)}")

fig = go.Figure()
vis.add_curve(y=y1, fig=fig)
vis.add_curve(y=y2+4, fig=fig)
for pair in pairings:
    i,j = pair
    fig.add_trace(go.Scatter(x=[i,j], y=[y1[i],y2[j]+4], line_color='rgba(150,150,150,0.5)', showlegend=False))
if DO_SHOW: fig.show()

# MORE REALISTIC CASE
x1 = np.arange(0, 10, 0.1)
y1 = np.cos(x1)*5

x2 = np.arange(0, 10, 0.3)
r2 = np.array([random() for _ in x2])
y2 = np.cos(x2+0.6)*5 + r2

score, p  = dynamic_time_warping(y1, y2)
print(f"DTW score in realistic case: {score:.2f}")
#print(p)

fig2 = go.Figure()
vis.add_curve(x=x1, y=y1, fig=fig2, name="Cosine")
vis.add_curve(x=x2, y=y2+4, fig=fig2, name="Noisy, shifted and undersampled cosine")
for pair in p:
    i,j = pair
    fig2.add_trace(go.Scatter(x=[x1[i],x2[j]], y=[y1[i],y2[j]+4], line_color='rgba(150,150,150,0.5)', showlegend=False))
if DO_SHOW: fig2.show()
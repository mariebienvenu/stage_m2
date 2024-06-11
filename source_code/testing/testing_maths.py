
import numpy as np
from random import random
import pandas as pd
import plotly.graph_objects as go

import app.visualisation as vis

DO_SHOW = True

## Test of derivatives

from app.maths_utils import derivee, derivee_seconde

step = 0.02
x = np.arange(0,5,step)
y = np.cos(x)

dy = derivee(y, step)
ddy = derivee_seconde(y, step)

fig = go.Figure()
vis.add_curve(y=y, x=x, fig=fig, name="y = cos(x)")
vis.add_curve(y=dy, x=x, fig=fig, name="y'(x) -> -sin(x)")
vis.add_curve(y=ddy, x=x, fig=fig, name="y''(x) -> -cos(x)")

if DO_SHOW: fig.show()

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
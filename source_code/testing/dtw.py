
import numpy as np
import random as rd
from random import random
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import app.visualisation as vis
from app.Curve import Curve
from app.DynamicTimeWarping import DynamicTimeWarping

DO_SHOW = True

# SIMPLE CASE: CONSTANT WARPING
y1 = np.array([1, 2, 3, 2, 3, 2, 1, 1])
y2 = np.array([1, 1, 2, 3, 2, 3, 2, 1])
co1 = np.vstack((list(range(y1.size)), y1)).T
co2 = np.vstack((list(range(y2.size)), y2)).T
curve1, curve2 = Curve(co1), Curve(co2)
dtw  = DynamicTimeWarping(curve1, curve2)
print(f"DTW score in ideal case: {dtw.score:.2f}")
print(f"Pairings found: {dtw.pairings}")
print(f"Computed table: \n{pd.DataFrame(dtw.DTW)}")

fig = go.Figure()
vis.add_curve(y=y1, fig=fig)
vis.add_curve(y=y2+4, fig=fig)
vis.add_pairings(y1=y1, y2=y2+4, pairs=dtw.pairings, fig=fig)
if DO_SHOW: fig.show()

# MORE REALISTIC CASE: NOISY COSINE
x1 = np.arange(0, 10, 0.1)
y1 = np.cos(x1)*5

x2 = np.arange(0, 10, 0.3)
r2 = np.array([random() for _ in x2])
y2 = np.cos(x2+0.6)*5 + r2

co1 = np.vstack((x1, y1)).T
co2 = np.vstack((x2, y2)).T
curve1, curve2 = Curve(co1), Curve(co2)

dtw  = DynamicTimeWarping(curve1, curve2)
score, p  = dtw.score, dtw.pairings
print(f"DTW score in realistic case: {score:.2f}")

fig2 = go.Figure()
vis.add_curve(x=x1, y=y1, fig=fig2, name="Cosine")
vis.add_curve(x=x2, y=y2+4, fig=fig2, name="Noisy, shifted and undersampled cosine")
vis.add_pairings(x1=x1, y1=y1, x2=x2, y2=y2+4, pairs=p, fig=fig2)
if DO_SHOW: fig2.show()

## LOCAL AND GLOBAL CONSTRAINTS: IMPULSIVE SIGNAL

rd.seed(2)

x1 = np.linspace(0, 10, 100)
y1 = np.array([random() for _ in x1])
y1[12], y1[24], y1[67] = 10, 20, 30

y2 = np.array([random() for _ in x1])
y2[6], y2[51], y2[96] = 10, 20, 30

co1 = np.vstack((x1, y1)).T
co2 = np.vstack((x1, y2)).T
curve1, curve2 = Curve(co1), Curve(co2)
curve1.normalize(), curve2.normalize()

dtw = DynamicTimeWarping(curve1, curve2)
constraints = dtw.local_constraints()
global_constraints = dtw.global_constraints()

fig3 = go.Figure()
vis.add_curve(x=x1, y=y1, fig=fig3, name="Curve1")
vis.add_curve(x=x1, y=y2+4, fig=fig3, name="Curve2")
vis.add_pairings(x1=x1, y1=y1, x2=x1, y2=y2+4, pairs=dtw.pairings, fig=fig3)
fig3.update_layout(title="DTW pairings - synthetic data")
if DO_SHOW: fig3.show()

fig4 = go.Figure()
vis.add_curve(constraints, style='lines', name="Local", fig=fig4)
vis.add_curve(global_constraints, style='lines', name="Global", fig=fig4)
fig4.update_layout(title="Local VS Global constraints on DTW - synthetic data")
if DO_SHOW: fig4.show()

fig5 = make_subplots(rows=2, cols=3, row_titles=["Pair location in cost matrix", "New distance propagation"], column_titles=["On spike", "Right next to spike", "Noisy area"])
for column, index in enumerate([12, 13, 85]):
    DTW = dtw.global_constraints_distances[index, :, :]
    ix, iy = np.array(dtw.pairings)[::-1][index]
    vis.add_heatmap(pd.DataFrame(dtw.cost_matrix), fig=fig5, row=1, col=column+1)
    fig5.add_vline(x=iy, line_color='rgb(1, 255, 1)', row=1, col=column+1, line_dash="dot")
    fig5.add_hline(y=ix, line_color='rgb(1, 255, 1)', row=1, col=column+1, line_dash="dot")
    DTW[DTW>15]=np.inf
    vis.add_heatmap(pd.DataFrame(DTW), min=0, max=15, fig=fig5, row=2, col=column+1)     
fig5.update_layout(title="Examples of global constraints on DTW - synthetic data")
if DO_SHOW: fig5.show()

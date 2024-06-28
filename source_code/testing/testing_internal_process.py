
import numpy as np
import pandas as pd
import random as rd
from random import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.internal_process import InternalProcess
import app.visualisation as vis
from app.curve import Curve
from app.animation import Animation
from app.color import Color

DO_SHOW = True

## Let's create some synthetic impulsive signals with no number issue for now

rd.seed(2)

x1 = np.linspace(0, 100, 101)
y1 = np.array([random() for _ in x1])
y1[12], y1[24], y1[67] = 10, 20, 30

y2 = np.array([random() for _ in x1])
y2[6], y2[51], y2[96] = 10, 20, 30

smoothing_kernel = [0.2, 0.6, 0.2] #[0.1, 0.2, 0.4, 0.2, 0.1] # 
y1 = np.convolve(y1, smoothing_kernel, "same")
y2 = np.convolve(y2, smoothing_kernel, "same")

co1 = np.vstack((x1, y1)).T
co2 = np.vstack((x1, y2)).T
vcurve1, vcurve2 = Curve(co1, fullname="Feature curve"), Curve(co2, fullname="Feature curve")


## And a matching animation curve

x3 = [0, 12, 18, 24, 45, 67, 100]
y3 = [3, 0, 2, 0, 1, 0, 0]
bcurve1 = Curve(np.vstack((x3, y3)).T, fullname="Animation Curve")

## And now, an InternalProcess object

print("CASE WHERE THERE IS NO MISMATCH BETWEEN REFERENCE AND TARGET CURVES")

internal = InternalProcess(Animation([vcurve1]), Animation([vcurve2]), Animation([bcurve1]), verbose=10)
banim2 = internal.process(feature="Feature curve", channels=["Animation Curve"], only_temporal=True, filter_indexes=True, blend_if_issue=False, normalize=True)

## Let's take a closer look

Color.reset()
fig0 = go.Figure() ## The curves and the result
for curve, name in zip([vcurve1, vcurve2, bcurve1, banim2[0]], ["Feature Ref", "Feature target", "Anim ref", "Anim result"]):
    curve.display(fig=fig0, handles=False, style="lines+markers", name=name)

if DO_SHOW: fig0.show()

# The matches between the feature curves with different degrees of selection

def visualise_matches(internal:InternalProcess):
    Color.reset()

    dtw = internal.dtw
    pairs = dtw.pairings
    constraints = internal.dtw_constraints

    naive_indexes = [0] + [index for index in range(1, len(pairs)-1) if constraints[index]>InternalProcess.COST_THRESHOLD] + [len(pairs)-1]
    refined_indexes = dtw.filtered_indexes()

    fig1 = make_subplots(
        rows=1, cols=3, 
        shared_xaxes='all', 
        subplot_titles=["No filter", "Basic filter (1 criteria)", "Refined filter  (several criteria)"],
        vertical_spacing=0.1, horizontal_spacing=0.1,
    )

    color1, color2 = Color.next(), Color.next()

    x1, y1, x2, y2 = dtw.times1, dtw.values1, dtw.times2, dtw.values2+4
    all_pairings = pairs
    naive_pairings = [e for i,e in enumerate(pairs) if i in naive_indexes]
    refined_pairings = [e for i,e in enumerate(pairs) if i in refined_indexes]
    for col in [1,2, 3]:
        vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=all_pairings, color=(210, 210, 210), opacity=1, fig=fig1, row=1, col=col)
        vis.add_curve(y=y2, x=x2, name="curve1", color=color2, fig=fig1, row=1, col=col)
        vis.add_curve(y=y1, x=x1, name="curve2", color=color1, fig=fig1, row=1, col=col)
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=all_pairings, color="green", opacity=1, fig=fig1, row=1, col=1)
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=naive_pairings, color="green", opacity=1, fig=fig1, row=1, col=2)
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=refined_pairings, color="green", opacity=1, fig=fig1, row=1, col=3)

    fig1.update_layout(
        xaxis1_title="Time (frames)",
        xaxis2_title="Time (frames)",
        xaxis3_title="Time (frames)",
        yaxis_title="Amplitude (arbitrary)",
    )
    title="ALL VS Basic VS Refined matches"
    fig1.update_layout(title=title)
    if DO_SHOW: fig1.show()
    return fig1

visualise_matches(internal)

## Visualise the inliers and outliers on the shortest path with DTW cost as a background

def visualise_graph(internal:InternalProcess):
    Color.reset()
    dtw = internal.dtw
    pairs = dtw.pairings
    inliers = internal.kept_indexes
    outliers = getattr(internal, "outliers", [])
    fig2 = vis.add_heatmap(pd.DataFrame(dtw.cost_matrix))
    inlier_color, outlier_color = Color.next(), Color.next()
    for index  in inliers:
        i,j = pairs[index][1]-pairs[0][1], pairs[index][0]-pairs[0][0]
        fig2.add_shape(
            type="circle",
            x0=i-2, y0=j-2, x1=i+2, y1=j+2,
            line_color=Color.to_string(inlier_color),
            name=f'{index}',
        )
    for index  in outliers:
        i,j = pairs[index][1]-pairs[0][1], pairs[index][0]-pairs[0][0]
        fig2.add_shape(
            type="circle",
            x0=i-2, y0=j-2, x1=i+2, y1=j+2,
            line_color=Color.to_string(outlier_color),
            name=f'{index}',
        )
    vis.add_curve(y=np.array(pairs)[:,0]-pairs[0][0], x=np.array(pairs)[:,1]-pairs[0][1], fig=fig2)
    title = "Shortest path across cost matrix"
    fig2.update_layout(title=title)
    if DO_SHOW: fig2.show()
    return fig2

visualise_graph(internal)

## Now, what about the case where there is a number issue ?
print("CASE WHERE THERE IS ONE TOO MANY IMPULSE IN THE REFERENCE CURVE")

y2 = np.array([random() for _ in x1])
y2[6], y2[51] = 10, 20
y2 = np.convolve(y2, smoothing_kernel, "same")

co2 = np.vstack((x1, y2)).T
new_vcurve2 = Curve(co2, fullname="Feature curve")

print("-- Without trying to fix the issue")
new_internal = InternalProcess(Animation([vcurve1]), Animation([new_vcurve2]), Animation([bcurve1]), verbose=10)
new_banim2 = new_internal.process(feature="Feature curve", channels=["Animation Curve"], only_temporal=True, filter_indexes=True, detect_issue=False, normalize=True)
new_curve1 = getattr(new_internal, "new_curve1", new_internal.curve1)

Color.reset()
fig2 = go.Figure() ## The curves and the result
for curve, name in zip([vcurve1, new_curve1, new_vcurve2, bcurve1, new_banim2[0]], ["Feature Ref", "Feature ref - kept", "Feature target", "Anim ref", "Anim result"]):
    curve.display(fig=fig2, handles=False, style="lines+markers", name=name)

if DO_SHOW: fig2.show()

visualise_matches(new_internal)

new_internal.detect_number_issues()
visualise_graph(new_internal)

## Let's try to fix this

print("-- Trying to fix the issue")
fixed_internal = InternalProcess(Animation([vcurve1]), Animation([new_vcurve2]), Animation([bcurve1]), verbose=10)
fixed_banim2 = fixed_internal.process(feature="Feature curve", channels=["Animation Curve"], only_temporal=True, filter_indexes=True, detect_issue=True, normalize=True)
fixed_curve1 = fixed_internal.new_curve1

Color.reset()
fig3 = go.Figure() ## The curves and the result
for curve, name in zip([fixed_internal.curve1, fixed_curve1, fixed_internal.curve2, bcurve1, fixed_banim2[0]], ["Feature Ref", "Feature ref - modified", "Feature target", "Anim ref", "Anim result"]):
    curve.display(fig=fig3, handles=False, style="lines+markers", name=name)
if DO_SHOW: fig3.show()

visualise_matches(fixed_internal)
fig4 = visualise_graph(fixed_internal)

constraints = fixed_internal.dtw_constraints
curves = [fixed_internal.curve1, fixed_curve1, fixed_internal.curve2, bcurve1, fixed_banim2[0]]
values = [curve.get_values() for curve in curves]

## Now what about the case where we have too few impulses instead of too many
print("CASE WHERE THERE IS TOO FEW IMPULSES (1 MISSING) IN THE REFERENCE CURVE")

x1 = np.linspace(0, 100, 101)
y1 = np.array([random() for _ in x1])
y1[12], y1[24], y1[67] = 40, 30, 20
y1 = np.convolve(y1, smoothing_kernel, "same")

y2 = np.array([random() for _ in x1])
y2[6], y2[51], y2[78], y2[96] = 40, 30, 20, 40
y2 = np.convolve(y2, smoothing_kernel, "same")

co1 = np.vstack((x1, y1)).T
co2 = np.vstack((x1, y2)).T
vcurve1 = Curve(co1, fullname="Feature curve")
new_vcurve2 = Curve(co2, fullname="Feature curve")

x3 = [0, 12, 18, 24, 45, 67, 100]
y3 = [3, 0, 2, 0, 1, 0, 0]
bcurve1 = Curve(np.vstack((x3, y3)).T, fullname="Animation Curve")

print("-- Without trying to fix the issue")
new_internal = InternalProcess(Animation([vcurve1]), Animation([new_vcurve2]), Animation([bcurve1]), verbose=10)
new_banim2 = new_internal.process(feature="Feature curve", channels=["Animation Curve"], only_temporal=True, filter_indexes=True, detect_issue=False, normalize=True)
new_curve1 = getattr(new_internal, "new_curve1", new_internal.curve1)

Color.reset()
fig2 = go.Figure() ## The curves and the result
for curve, name in zip([vcurve1, new_curve1, new_vcurve2, bcurve1, new_banim2[0]], ["Feature Ref", "Feature ref - kept", "Feature target", "Anim ref", "Anim result"]):
    curve.display(fig=fig2, handles=False, style="lines+markers", name=name)

if DO_SHOW: fig2.show()

visualise_matches(new_internal)

new_internal.detect_number_issues()
visualise_graph(new_internal)

## Let's try to fix this

print("-- Trying to fix the issue")
fixed_internal = InternalProcess(Animation([vcurve1]), Animation([new_vcurve2]), Animation([bcurve1]), verbose=10)
fixed_banim2 = fixed_internal.process(feature="Feature curve", channels=["Animation Curve"], only_temporal=True, filter_indexes=True, detect_issue=True, normalize=True)
fixed_curve1 = fixed_internal.new_curve1

Color.reset()
fig3 = go.Figure() ## The curves and the result
for curve, name in zip([fixed_internal.curve1, fixed_curve1, fixed_internal.curve2, bcurve1, fixed_banim2[0]], ["Feature Ref", "Feature ref - modified", "Feature target", "Anim ref", "Anim result"]):
    curve.display(fig=fig3, handles=False, style="lines+markers", name=name)
if DO_SHOW: fig3.show()

visualise_matches(fixed_internal)
fig5 = visualise_graph(fixed_internal)
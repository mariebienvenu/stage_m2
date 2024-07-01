
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

figures, titles = internal.make_diagrams(number_issues=False)
for fig, title in zip(figures, titles):
    fig.update_layout(title=title)
    if DO_SHOW: fig.show()

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

figures, titles = new_internal.make_diagrams(number_issues=False)
for fig, title in zip(figures, titles):
    fig.update_layout(title=title)
    if DO_SHOW: fig.show()

new_internal.detect_number_issues()

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

figures, titles = fixed_internal.make_diagrams(number_issues=True)
for fig, title in zip(figures, titles):
    fig.update_layout(title=title)
    if DO_SHOW: fig.show()

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

figures, titles = new_internal.make_diagrams(number_issues=False)
for fig, title in zip(figures, titles):
    fig.update_layout(title=title)
    if DO_SHOW: fig.show()

new_internal.detect_number_issues()

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

figures, titles = fixed_internal.make_diagrams(number_issues=True)
for fig, title in zip(figures, titles):
    fig.update_layout(title=title)
    if DO_SHOW: fig.show()

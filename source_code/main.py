
import os, sys

def check_sys_path(
        packages_path = "C:/Users/Marie Bienvenu/miniconda3/envs/blender2/Lib/site-packages",
        source_code_path = "C:/Users/Marie Bienvenu/stage_m2/source_code/"
    ):
    if packages_path not in sys.path:
         sys.path.append(packages_path)  # removes package import errors
    if source_code_path not in sys.path:
         sys.path.append(source_code_path)  # removes local import errors

check_sys_path()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
 
import app.main as main
import app.visualisation as vis
import app.dynamic_time_warping as DTW
import app.abstract_io, app.internal_process, app.warping, app.dcc_io, app.blender_utils, app.video_io
import app.animation, app.curve, app.color

import importlib
importlib.reload(main)
importlib.reload(app.abstract_io)
importlib.reload(app.internal_process)
importlib.reload(app.warping)
importlib.reload(DTW)
importlib.reload(app.dcc_io)
importlib.reload(app.blender_utils)
importlib.reload(app.video_io)
importlib.reload(app.animation)
importlib.reload(app.curve)
importlib.reload(app.color)
importlib.reload(vis)

Color = app.color.Color
warping = app.warping
Color.reset()
Color.next()

possible_features = ["Velocity Y", "First derivative of Velocity Y", "Location Y", "First derivative of Location Y", "Second derivative of Location Y"]
chosen_feature = possible_features[-1]
## il se trouve que c'est Location Y qui est la meilleure in fine sur le score, mais Second derivative of Location Y gère mieux le timing des rebonds

directory = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes/bouncing_ball_retiming/"
main_obj = main.Main(directory, verbose=2)
feature = main_obj.connexions_of_interest[0]["video feature"]
main_obj.connexions_of_interest[0]["video feature"] = chosen_feature

main_obj.process(force=True)
main_obj.to_blender()
main_obj.blender_scene.from_software() #to recover new fcurve pointer

main_obj.draw_diagrams(show=True)

curve_feature = main_obj.internals[0].vanim1.find(chosen_feature)

blender_anim = main_obj.blender_scene.get_animations()[0]
blender_start, blender_stop = blender_anim.time_range
blender_anim_sampled = blender_anim.sample(n_samples=blender_stop-blender_start+1, start=blender_start, stop=blender_stop)
blender_anim_sampled.enrich()

curve_blender = blender_anim_sampled.find("Second derivative of location Z sampled")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
curve_blender.display(handles=False, style="lines", fig=fig, row=1, col=1)
curve_feature.display(handles=False, style="lines", fig=fig, row=2, col=1)
fig.show()

from app.maths_utils import correlation

def get_offset_correlation(curve1:app.curve.Curve, curve2:app.curve.Curve, offset_max=10):
    # curves have to be sampled with the same timestep but not necessarily with exactly the same time range
    offsets = list(range(-offset_max, offset_max+1))
    correlations=np.zeros((len(offsets)))
    (start1, stop1), (start2, stop2) = curve1.time_range, curve2.time_range
    for i,offset in enumerate(offsets):
        start = max(start1+offset, start2)
        stop = min(stop1+offset, stop2)
        values1 = [val for (val,time) in zip(curve1.get_values(),curve1.get_times()) if time+offset<=stop and time+offset>=start]
        values2 = [val for (val,time) in zip(curve2.get_values(),curve2.get_times()) if time<=stop and time>=start]
        assert len(values1)==len(values2), "Problem"
        correl = correlation(values1, values2)
        correlations[i] = correl
    correlations = np.abs(correlations)
    return offsets, correlations

def get_best_offset(offsets, correlations, correlation_threshold=0.5):
    best_correl = np.max(correlations)
    print(best_correl)
    if best_correl < correlation_threshold:
        return 0 # best offset found not convincing enough
    return offsets[np.argmax(correlations)]

matrix_offset = np.zeros((len(main_obj.internals[0].vanim1),len(blender_anim_sampled)))
matrix_correlations = np.zeros((len(main_obj.internals[0].vanim1),len(blender_anim_sampled)))
for i,curve1 in enumerate(main_obj.internals[0].vanim1):
    for j,curve2 in enumerate(blender_anim_sampled):
        offsets, correlations = get_offset_correlation(curve1, curve2)
        matrix_offset[i,j] = get_best_offset(offsets, correlations)
        matrix_correlations[i,j]=np.max(correlations)
        #print(f"Curve1:{curve1.fullname}, Curve2:{curve2.fullname}, best offset:{get_best_offset(curve1, curve2)}")
print(matrix_correlations)
df = pd.DataFrame(matrix_correlations)
vis.add_heatmap(df, doShow=True)
print(main_obj.internals[0].vanim1)
print(blender_anim_sampled)
#print(get_best_offset(curve_feature, curve_blender)) # expect 3
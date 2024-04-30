
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
 
import app.Main as main
import app.visualisation as vis

import importlib
importlib.reload(main)
importlib.reload(main.absIO)
importlib.reload(main.InternalProcess)
importlib.reload(main.InternalProcess.Warp)
importlib.reload(main.InternalProcess.DynamicTimeWarping)
importlib.reload(main.SoftIO)
importlib.reload(main.SoftIO.b_utils)
importlib.reload(main.VideoIO)
importlib.reload(main.VideoIO.Animation)
importlib.reload(main.VideoIO.Animation.Curve)
importlib.reload(vis)

Color = main.VideoIO.Animation.Curve.Color
Color.reset()
Color.next()

possible_features = ["Velocity Y", "First derivative of Velocity Y", "Location Y", "First derivative of Location Y", "Second derivative of Location Y"]
## il se trouve que c'est Location Y qui est la meilleure in fine sur le score, mais Second derivative of Location Y gère mieux le timing des rebonds

directory = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes/bouncing_ball_x2/"
main_obj = main.Main(directory, verbose=2)
feature = main_obj.connexions_of_interest[0]["video feature"]

main_obj.process(force=True)
main_obj.to_blender()
main_obj.blender_scene.from_software() #to recover new fcurve pointer

og_anim = main_obj.blender_scene.original_anims[0]
edited_anim = main_obj.blender_scene.get_animations()[0] # will have an fcurve pointer, unlike main_obj.new_anims[0] -> does not work as expected
edited_anim = main.SoftIO.b_utils.get_animation("Ball_edited") # to force fcurve pointer retrieval

edited_curve, og_curve = edited_anim.find("location Z"), og_anim.find("location Z")
og_curve.rename("Ground truth")
edited_curve.color = Color.next()

gt_sampled = og_curve.sample(list(range(int(og_curve.time_range[0]),int(og_curve.time_range[1])+1)))
gt_sampled.time_scale(1, 0.5)
edited_sampled = edited_curve.sample(list(range(int(edited_curve.time_range[0]),int(edited_curve.time_range[1])+1)))

og_times = og_curve.get_times()
edited_times = edited_curve.get_times() # works !
gt_times = (og_times-1)/2 + 1

fig = make_subplots(rows=1, cols=2, column_titles=["Curves", "Time Warp"])
vis.add_curve(edited_times, x=og_times, fig=fig, row=1, col=2, name="Warping effect on keyframes", color=f'rgb{edited_curve.color}')
vis.add_curve(gt_times, x=og_times, fig=fig, row=1, col=2, name="Ground truth", color=f'rgb{og_curve.color}')
#og_anim.display(fig=fig, row=1, col=1)
#edited_anim.display(fig=fig, row=1, col=1)
gt_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines", name="Ground truth")
edited_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines", name="Edited curve")
fig.show()

og_sampled = og_curve.sample(list(range(int(og_curve.time_range[0]),int(og_curve.time_range[1])+1)))
og_sampled.color = og_curve.color
fig2 = make_subplots(rows=2, shared_xaxes=True, row_titles=["Original animation curve", "Edited animation curve"])
fig2.update_layout(title_text=f"Editing using '{feature}' video feature")
og_curve.display(fig=fig2, row=1, col=1)
edited_curve.display(fig=fig2, row=2, col=1)
og_sampled.display(fig=fig2, row=1, col=1, handles=False, style="lines")
edited_sampled.display(fig=fig2, row=2, col=1, handles=False, style="lines")
fig2.show()

offset = 2
fig3 = go.Figure()
internal = main_obj.internals[0]
x1, y1, x2, y2 = internal.dtw.times1, internal.dtw.values1, internal.dtw.times2, internal.dtw.values2
vis.add_curve(x=x1, y=y1, fig=fig3, name="Reference")
vis.add_curve(x=x2, y=y2+offset, fig=fig3, name="Target")
vis.add_pairings(x1=x1, y1=y1, x2=x2, y2=y2+offset, pairs=internal.dtw.pairings, color='rgba(150,150,150,0.5)', fig=fig3)
fig3.update_layout(title_text=f"Dynamic Time Warping using '{feature}' video feature")
fig3.show()

## DTW array terrain
fig4 = go.Figure()
value_ref = internal.dtw.values1
value_target = internal.dtw.values2
time_ref, time_target = internal.dtw.bijection
df = pd.DataFrame(internal.dtw.cost_matrix)
vis.add_heatmap(df, fig=fig4)
#vis.add_curve(y=time_ref, x=time_target, fig=fig4)
fig4.show()
 
# not working
fig5 = go.Figure()
fig5.add_trace(go.Surface(z=df))
fig5.show(renderer='browser')

# DTW cumulative array terrain
df2 = pd.DataFrame(internal.dtw.DTW)
fig6 = vis.add_heatmap(df2)
fig6.show()
 
# DTW cumulative array terrain without too costly pixels
dtw_array = np.copy(internal.dtw.DTW)
dtw_array[dtw_array>internal.dtw.score] = np.inf
df3 = pd.DataFrame(dtw_array)
fig7 = vis.add_heatmap(df3)
fig7.show()

# Measure of constraint of shortest path on DTW array terrain -> the more constrained, the more certain/trustworthy
w_size = 10
local_constraints = internal.measure_dtw_local_constraint(w_size)
fig8 = vis.add_curve(y=local_constraints)
fig8.show()


# To compare : constraint in the case of the location feature instead of acceleration
main_obj2 = main.Main(directory, verbose=2)
main_obj2.connexions_of_interest[0]["video feature"] = "Location Y"

main_obj2.process(force=True)
main_obj2.to_blender()
main_obj2.blender_scene.from_software() #to recover new fcurve pointer
internal2 = main_obj2.internals[0]

df3 = pd.DataFrame(internal2.dtw.cost_matrix)
fig11 = vis.add_heatmap(df3, min=0, max=np.max(internal.dtw.cost_matrix))
fig11.show()

local_constraints2 = internal2.measure_dtw_local_constraint(w_size)
fig9 = vis.add_curve(y=local_constraints2)
fig9.show()

fig10 = vis.add_curve(y=local_constraints)
vis.add_curve(y=local_constraints2, fig=fig10)
fig10.show()

## -> il semblerait qu'un threshold de 1 sur le niveau de contrainte permette de trouver les points de timing important portés par un signal impulsionnel !

constraints_acc = internal.measure_dtw_local_constraint(w_size)
constraints_loc = internal2.measure_dtw_local_constraint(w_size)
fig12 = go.Figure()
vis.add_curve(y=constraints_acc, name='Feature "Acceleration"', fig=fig12)
vis.add_curve(y=constraints_loc, name='Feature "Location"', fig=fig12)
fig12.show()

fig13 = go.Figure()
ws = [3, 5, 7, 10, 15, 20, 35, 50]
constraints = [internal.measure_dtw_local_constraint(w) for w in ws] 
for w, constraint in zip(ws, constraints):
    vis.add_curve(y=constraint, name=w, fig=fig13)
fig13.show()

## Now with simplified warp
main_obj3 = main.Main(directory, verbose=2)
main_obj3.connexions_of_interest[0]["video feature"] = "Second derivative of Location Y"

main_obj3.process(force=True)
main_obj3.to_blender()
main_obj3.blender_scene.from_software() #to recover new fcurve pointer
internal3 = main_obj3.internals[0]
print(internal3.dtw.bijection)
a=1



og_anim = main_obj3.blender_scene.original_anims[0]
edited_anim = main.SoftIO.b_utils.get_animation("Ball_edited") # to force fcurve pointer retrieval

edited_curve, og_curve = edited_anim.find("location Z"), og_anim.find("location Z")
og_curve.rename("Ground truth")
edited_curve.color = Color.next()

gt_sampled = og_curve.sample(list(range(int(og_curve.time_range[0]),int(og_curve.time_range[1])+1)))
gt_sampled.time_scale(1, 0.5)
edited_sampled = edited_curve.sample(list(range(int(edited_curve.time_range[0]),int(edited_curve.time_range[1])+1)))

og_times = og_curve.get_times()
edited_times = edited_curve.get_times() # works !
gt_times = (og_times-1)/2 + 1

fig = make_subplots(rows=1, cols=2, column_titles=["Curves", "Time Warp"])
vis.add_curve(edited_times, x=og_times, fig=fig, row=1, col=2, name="Experimental", color=f'rgb{edited_curve.color}')
vis.add_curve(gt_times, x=og_times, fig=fig, row=1, col=2, name="Ground truth", color=f'rgb{og_curve.color}')
#og_anim.display(fig=fig, row=1, col=1)
#edited_anim.display(fig=fig, row=1, col=1)
gt_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines")
edited_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines")

fig.show()

## TODO main - clean the script...
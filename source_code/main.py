
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
importlib.reload(main.InternalProcess.m_utils)
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
vis.add_curve(edited_times, x=og_times, fig=fig, row=1, col=2, name="Experimental", color=f'rgb{edited_curve.color}')
vis.add_curve(gt_times, x=og_times, fig=fig, row=1, col=2, name="Ground truth", color=f'rgb{og_curve.color}')
#og_anim.display(fig=fig, row=1, col=1)
#edited_anim.display(fig=fig, row=1, col=1)
gt_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines")
edited_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines")
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
x1, y1, x2, y2 = internal.warp_time_ref, internal.warp_value_ref, internal.warp_time_target, internal.warp_value_target
vis.add_curve(x=x1, y=y1, fig=fig3, name="Reference")
vis.add_curve(x=x2, y=y2+offset, fig=fig3, name="Target")
vis.add_pairings(x1=x1, y1=y1, x2=x2, y2=y2+offset, pairs=internal.dtw_pairings, color='rgba(150,150,150,0.5)', fig=fig3)
fig3.update_layout(title_text=f"Dynamic Time Warping using '{feature}' video feature")
fig3.show()

## DTW array terrain
fig4 = go.Figure()
value_ref = internal.warp_value_ref
value_target = internal.warp_value_target
time_ref, time_target = internal.x_in, internal.x_out
cost_matrix = np.array([[abs(value_ref[i] - value_target[j]) for j in range(value_target.size)] for i in range(value_ref.size)])
df = pd.DataFrame(cost_matrix)
vis.add_heatmap(df, fig=fig4)
#vis.add_curve(y=time_ref, x=time_target, fig=fig4)
fig4.show()
 
# not working
fig5 = go.Figure()
fig5.add_trace(go.Surface(z=df))
fig5.show(renderer='browser')

# DTW cumulative array terrain
df2 = pd.DataFrame(internal.dtw_array)
fig6 = vis.add_heatmap(df2)
fig6.show()
 
# DTW cumulative array terrain without too costly pixels
dtw_array = np.copy(internal.dtw_array)
dtw_array[dtw_array>internal.dtw_score] = np.inf
df3 = pd.DataFrame(dtw_array)
fig7 = vis.add_heatmap(df3)
fig7.show()

# Measure of constraint of shortest path on DTW array terrain -> the more constrained, the more certain/trustworthy
w_size = 10
time_ref, time_target = np.array(time_ref), np.array(time_target)
local_constraints = [0 for _ in range(time_ref.size)]
for i in range(2*w_size, time_ref.size-3*w_size):
    time_x = int(time_ref[i])
    time_y = int(time_target[i])
    center_cost = cost_matrix[time_x, time_y]
    upper_costs = cost_matrix[time_x+1:time_x+w_size, time_y] + cost_matrix[time_x, time_y+1:time_y+w_size]
    lower_costs = cost_matrix[time_x-w_size:time_x, time_y] + cost_matrix[time_x, time_y-w_size:time_y]
    alternative_costs = np.concatenate((upper_costs, lower_costs)) #[patch[j,k] for j in range(patch.shape[0]) for k in range(patch.shape[1]) if (j!=w_size or k!=w_size)]
    minimal_additionnal_cost = np.min(alternative_costs) - center_cost
    local_constraints[i] = max(0,minimal_additionnal_cost)
fig8 = vis.add_curve(y=local_constraints)
fig8.show()


# To compare : constraint in the case of the location feature instead of acceleration
main_obj2 = main.Main(directory, verbose=2)
main_obj2.connexions_of_interest[0]["video feature"] = "Location Y"

main_obj2.process(force=True)
main_obj2.to_blender()
main_obj2.blender_scene.from_software() #to recover new fcurve pointer
internal2 = main_obj2.internals[0]
value_ref2 = internal2.warp_value_ref
value_target2 = internal2.warp_value_target
time_ref2, time_target2 = internal2.x_in, internal2.x_out
cost_matrix2 = np.array([[abs(value_ref2[i] - value_target2[j]) for j in range(value_target2.size)] for i in range(value_ref2.size)])

df3 = pd.DataFrame(cost_matrix2)
fig11 = vis.add_heatmap(df3, min=0, max=np.max(cost_matrix))
fig11.show()

time_ref2, time_target2 = np.array(time_ref2), np.array(time_target2)
local_constraints2 = [0 for _ in range(time_ref2.size)]
for i in range(2*w_size, time_ref2.size-3*w_size):
    time_x = int(time_ref2[i])
    time_y = int(time_target2[i])
    center_cost = cost_matrix2[time_x, time_y]
    upper_costs = cost_matrix2[time_x+1:time_x+w_size, time_y] + cost_matrix2[time_x, time_y+1:time_y+w_size]
    lower_costs = cost_matrix2[time_x-w_size:time_x, time_y] + cost_matrix2[time_x, time_y-w_size:time_y]
    alternative_costs = np.concatenate((upper_costs, lower_costs))
    minimal_additionnal_cost = np.min(alternative_costs) - center_cost
    local_constraints2[i] = max(0,minimal_additionnal_cost)
fig9 = vis.add_curve(y=local_constraints2)
fig9.show()

fig10 = vis.add_curve(y=local_constraints)
vis.add_curve(y=local_constraints2, fig=fig10)
fig10.show()

## -> il semblerait qu'un threshold de 1 sur le niveau de contrainte permette de trouver les points de timing important portés par un signal impulsionnel !

def measure_dtw_local_constraint(internal:main.InternalProcess.InternalProcess, window_size=10):

    value_ref = np.copy(internal.warp_value_ref)
    value_target = np.copy(internal.warp_value_target)
    cost_matrix = np.array([[abs(value_ref[i] - value_target[j]) for j in range(value_target.size)] for i in range(value_ref.size)]) # shape (N_in, N_out)
    
    time_ref, time_target = np.copy(np.array(internal.x_in)).astype(int), np.copy(np.array(internal.x_out)).astype(int) # DTW pairs ; two arrays of same size ; dimension: frame number
    timerange_ref, timerange_target = time_ref[-1]-time_ref[0], time_target[-1]-time_target[0]
    N = time_ref.size

    local_constraints = np.zeros((N))
    #debug = np.zeros((N), dtype=int)
    for i in range(1, N-1):
        index_x, index_y = time_ref[i]-time_ref[0], time_target[i]-time_target[0] # DTW pair number i back in index space, for cost_matrix
        center_cost = cost_matrix[index_x, index_y] # best cost (globally)

        w_size = min(window_size, index_x, index_y, timerange_ref-index_x, timerange_target-index_y)
        #debug[i] = w_size
        upper_costs = cost_matrix[index_x+1:index_x+w_size, index_y] + cost_matrix[index_x, index_y+1:index_y+w_size]
        lower_costs = cost_matrix[index_x-w_size:index_x, index_y] + cost_matrix[index_x, index_y-w_size:index_y]
        alternative_costs = np.concatenate((upper_costs, lower_costs))
        minimal_additionnal_cost = np.min(alternative_costs) - center_cost if alternative_costs.size>0 else 0
        local_constraints[i] = max(0,minimal_additionnal_cost)

    return local_constraints

constraints_acc = measure_dtw_local_constraint(internal)
constraints_loc = measure_dtw_local_constraint(internal2)
fig12 = go.Figure()
vis.add_curve(y=constraints_acc, name='Feature "Acceleration"', fig=fig12)
vis.add_curve(y=constraints_loc, name='Feature "Location"', fig=fig12)
fig12.show()

fig13 = go.Figure()
ws = [3, 5, 7, 10, 15, 20, 35, 50]
constraints = [measure_dtw_local_constraint(internal, w) for w in ws] 
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
print(internal3.x_in, internal3.x_out)
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
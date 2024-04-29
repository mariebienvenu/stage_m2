
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
 
import app.Main as main
import app.visualisation as vis

import importlib
importlib.reload(main)
importlib.reload(main.absIO)
importlib.reload(main.InternalProcess)
importlib.reload(main.InternalProcess.Warp)
importlib.reload(main.SoftIO)
importlib.reload(main.SoftIO.b_utils)
importlib.reload(main.VideoIO)
importlib.reload(main.VideoIO.Animation)
importlib.reload(main.VideoIO.Animation.Curve)
importlib.reload(vis)

Color = main.VideoIO.Animation.Curve.Color
Color.reset()

possible_features = ["Velocity Y", "First derivative of Velocity Y", "Location Y", "First derivative of Location Y", "Second derivative of Location Y"]
## il se trouve que c'est location Y qui est la meilleure in fine

directory = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes/bouncing_ball_x2/"
main_obj = main.Main(directory, verbose=2)

fig = make_subplots(rows=1, cols=2, column_titles=["Curves", "Time Warp"])

for i,feature in enumerate(possible_features):
    print(f"For feature {feature}:")
    main_obj.connexions_of_interest[0]["video feature"] = feature
    main_obj.process(force=True)
    main_obj.to_blender()
    main_obj.blender_scene.from_software() #to recover new fcurve pointer

    og_anim = main_obj.blender_scene.original_anims[0]
    edited_anim = main_obj.blender_scene.get_animations()[0] # will have an fcurve pointer, unlike main_obj.new_anims[0] -> does not work as expected
    edited_anim = main.SoftIO.b_utils.get_animation("Ball_edited") # to force fcurve pointer retrieval

    edited_curve, og_curve = edited_anim.find("location Z"), og_anim.find("location Z")
    edited_curve.rename(f"location Z -  {feature}")
    og_curve.rename("Ground truth")
    edited_curve.color = Color.next()

    gt_sampled = og_curve.sample(list(range(int(og_curve.time_range[0]),int(og_curve.time_range[1])+1)))
    gt_sampled.time_scale(1, 0.5)
    edited_sampled = edited_curve.sample(list(range(int(edited_curve.time_range[0]),int(edited_curve.time_range[1])+1)))

    og_times = og_curve.get_times()
    edited_times = edited_curve.get_times() # works !
    gt_times = (og_times-1)/2 + 1

    vis.add_curve(edited_times, x=og_times, fig=fig, row=1, col=2, name=feature, color=f'rgb{edited_curve.color}')
    if i==0: vis.add_curve(gt_times, x=og_times, fig=fig, row=1, col=2, name="Ground truth", color=f'rgb{og_curve.color}')
    #if i==0: og_anim.display(fig=fig, row=1, col=1)
    #edited_anim.display(fig=fig, row=1, col=1)
    if i==0: gt_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines")
    edited_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines")

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

fig.show()

'''
fig = vis.add_curve(edited_times, x=og_times)
vis.add_curve(gt_times, x=og_times, fig=fig)
fig.show()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_titles=["Original Animation", "Edited animation"])
og_anim.display(fig=fig, row=1, col=1)
edited_anim.display(fig=fig, row=2, col=1)
og_sampled.display(fig=fig, row=1, col=1, handles=False, style="lines")
edited_sampled.display(fig=fig, row=2, col=1, handles=False, style="lines")
fig.show()'''

## TODO - add visualisation of the DTW graph terrain
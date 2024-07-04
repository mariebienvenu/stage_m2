
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
import app.abstract_io, app.warping, app.dcc_io, app.blender_utils, app.video_io
import app.animation, app.color

"""
import app.blender_utils as b_utils
import app.curve as C
import app.internal_process as ip
import importlib
importlib.reload(b_utils)
importlib.reload(C)
importlib.reload(ip)

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
"""

Color = app.color.Color
warping = app.warping
Color.reset()
Color.next()

directory = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes/bouncing_ball_plus1/"
main_obj = main.Main(directory, verbose=10)

main_obj.process(force=True)
main_obj.to_blender()
main_obj.blender_scene.from_software() #to recover new fcurve pointer

main_obj.draw_diagrams(show=True)
figures, titles = main_obj.internals[0].make_diagrams(anim_style="lines+markers")
for fig, title in zip(figures, titles):
    fig.update_layout(title=title)
    fig.show()
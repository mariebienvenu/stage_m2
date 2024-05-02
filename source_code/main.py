
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

directory = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes/bouncing_ball_retiming/"
main_obj = main.Main(directory, verbose=2)
feature = main_obj.connexions_of_interest[0]["video feature"]
main_obj.connexions_of_interest[0]["video feature"] = "Second derivative of Location Y"

main_obj.process(force=True)
main_obj.to_blender()
main_obj.blender_scene.from_software() #to recover new fcurve pointer

main_obj.draw_diagrams(show=True)
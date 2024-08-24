
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
import app.internal_process as internpro
import app.visualisation as vis
import app.dynamic_time_warping as DTW
import app.abstract_io, app.warping, app.dcc_io, app.video_io
import app.blender_utils as b_utils
import app.animation, app.color

import importlib
importlib.reload(main)
importlib.reload(internpro)

Color = app.color.Color
warping = app.warping
Color.reset()
Color.next()    

BLENDER = True
#main.Main.SPOT_FOR_DTW_CONSTRAINTS = 1 #bugs
#internpro.InternalProcess.CONSTRAINT_THRESHOLD = 2.2 # walk_cycle
main.Main.USE_SEMANTIC = 5 #5 #bouncing ball test

PATH = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes"

directory = "/bouncing_ball_plus1_pretty/"
main_obj = main.Main(PATH+directory, verbose=10, no_blender=(not BLENDER))

main_obj.process(force=True)
if BLENDER:
    main_obj.to_blender()
    main_obj.blender_scene.from_software(in_place=True) #to recover new fcurve pointer
main_obj.display(show=False)

if directory == "/bouncing_ball_samenumber/":
    main.for_the_paper_workflow(main_obj)

if directory == "/bouncing_ball_plus1_pretty/":
    main.for_the_paper_teaser_and_ball(main_obj)

print("--------- END -----------")
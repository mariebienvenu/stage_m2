
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

directory = "C:/Users/Marie Bienvenu/stage_m2/afac/main/"
main_obj = main.Main(directory, verbose=2)
main_obj.process()

print(main_obj.blender_scene._animations[0].find("location Y").get_times())
print(main_obj.new_anims[0].find("location Y").get_times()) # works !

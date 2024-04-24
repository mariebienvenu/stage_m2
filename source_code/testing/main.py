
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

import app.Main as main

import importlib
importlib.reload(main)
importlib.reload(main.InternalProcess)
importlib.reload(main.InternalProcess.Warp)
importlib.reload(main.SoftIO)
importlib.reload(main.SoftIO.b_utils)
importlib.reload(main.VideoIO)
importlib.reload(main.VideoIO.Animation)
importlib.reload(main.VideoIO.Animation.Curve)

directory = "C:/Users/Marie Bienvenu/stage_m2/afac/main/"
main_obj = main.Main(directory, verbose=2)
main_obj.process()

print(main_obj.blender_scene._animations[0].find("location Y").get_times())
print(main_obj.new_anims[0].find("location Y").get_times()) # works !

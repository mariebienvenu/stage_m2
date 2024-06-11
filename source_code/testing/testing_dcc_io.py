import os, sys
import numpy as np
import plotly.graph_objects as go

def check_sys_path(
        packages_path = "C:/Users/Marie Bienvenu/miniconda3/envs/blender2/Lib/site-packages",
        source_code_path = "C:/Users/Marie Bienvenu/stage_m2/source_code/"
    ):
    if packages_path not in sys.path:
         sys.path.append(packages_path)  # removes package import errors
    if source_code_path not in sys.path:
         sys.path.append(source_code_path)  # removes local import errors

check_sys_path()

import app.dcc_io as dio

import importlib
importlib.reload(dio)


directory = "C:/Users/Marie Bienvenu/stage_m2/afac/"
softIO = dio.SoftIO(directory, verbose=2)
softIO.process() # will fetch the animation from the software

fig = go.Figure()
anims = softIO.get_animations()
for anim in anims:
    anim.display(fig=fig)
    anim.sample(100).display(handles=False, style="lines", fig=fig)
fig.show()


anims[0].find("location X").set_keyframe_attribute(1, "value", np.float64(5.)) # we changed the anim, but with no effect on blender. consequently, the sampling is incorrect.


fig2 = go.Figure()
anims = softIO.get_animations()
for anim in anims:
    anim.display(fig=fig2)
    anim.sample(100).display(handles=False, style="lines", fig=fig2)
fig2.show()

softIO.set_animations(anims) # will affect blender, which handles the sampling, so now the sampling should be correct.
# could have also used the softIO.to_software() method which mirror the softIO.from_software() used in softIO.process()

fig2 = go.Figure()
anims = softIO.get_animations()
for anim in anims:
    anim.display(fig=fig2)
    anim.sample(100).display(handles=False, style="lines", fig=fig2)
fig2.show()



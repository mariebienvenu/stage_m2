
import os
import numpy as np
from plotly.subplots import make_subplots

import app.Animation as Animation

data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

subdirectory = '04-08 appareil de Damien'
video_reference = 'P1010236'
video_target = 'P1010236_x2'

ref_movement = Animation.Animation().load(f'{data_path}/{subdirectory}/{video_reference}/')
ref_frame_times = ref_movement[0].get_times()
additionnal = [curve.first_derivative() for curve in ref_movement]+[curve.second_derivative() for curve in ref_movement]
additionnal_curves_ref = Animation.Animation(
    [curve for curve in additionnal if curve.get_value_range()[1]-curve.get_value_range()[0]>1e-2]
)
additionnal_curves_ref.display(handles=False, style='markers+lines', doShow=True)


target_movement = Animation.Animation().load(f'{data_path}/{subdirectory}/{video_target}/')
target_frame_times = target_movement[0].get_times()
additionnal = [curve.first_derivative() for curve in target_movement]+[curve.second_derivative() for curve in target_movement]
additionnal_curves_target = Animation.Animation(
    [curve for curve in additionnal if curve.get_value_range()[1]-curve.get_value_range()[0]>1e-2]
)
additionnal_curves_target.display(handles=False, style='markers+lines', doShow=True)

chosen_channel = 'First derivative of Velocity Y'

ref = (ref_movement+additionnal_curves_ref).find(chosen_channel)
ref.rename(f'{chosen_channel} - reference')
target = (target_movement+additionnal_curves_target).find(chosen_channel)
target.rename(f'{chosen_channel} - target')

fig = make_subplots(rows=2, cols=1, subplot_titles=["Reference", "Target"])
ref.display(fig=fig, handles=False, style='lines+markers', col=1, row=1)
target.display(fig=fig, handles=False, style='lines+markers', col=1, row=2)
fig.show()
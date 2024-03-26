import numpy as np

import app.blender_utils as b_utils
from app.Animation import Animation
from app.Curve import Curve

times = np.arange(0,10,1)
values = np.array([5, 5, 5, 10, 5, -10, -5, 0, 0, 0])
coordinates = np.vstack((times, values)).T

left_handle_x = np.expand_dims(np.arange(-0.3, 9.3, 1), axis=1)
right_handle_x = np.expand_dims(np.arange(0.3, 10.3, 1), axis=1)
left_handle_y = np.expand_dims(np.array([5, 5, 5, 10, 8, -10, -7, 0, 0, 0]), axis=1)
right_handle_y = np.expand_dims(np.array([5, 5, 5, 10, 2, -10, -3, 0, 0, 0]), axis=1)

fake_blender_curve = Curve(
    coordinates,
    tangent_left_handle_x=left_handle_x,
    tangent_left_handle_y=left_handle_y,
    tangent_right_handle_x=right_handle_x,
    tangent_right_handle_y=right_handle_y
)

## Test de b_utils.get_crop()

start, stop = b_utils.get_crop(fake_blender_curve)

fig = fake_blender_curve.display(style='markers+lines')
fig.add_vline(x=start, annotation_text="Start", annotation_position="top right")
fig.add_vline(x=stop, annotation_text="Start", annotation_position="top right")

fig.show()


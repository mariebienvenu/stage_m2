import numpy as np
from random import random
from plotly.subplots import make_subplots

from app.curve import Curve
from app.color import Color
from app.animation import Animation
import app.visualisation as vis

class FakeCurvePointer:

    def __init__(self, times=None, values=None):
        self.times = times
        self.values = values
        # assuming ordered times

    def evaluate(self, time):
        if self.times is None or self.values is None: return np.nan # None avant -> peut-être cassé quelque chose... 
        for t,v in zip(self.times, self.values):
            if t>=time:
                return v
            
DO_SHOW = True

# COLOR
            
for _ in range(20):
    color = Color.next()
    #print(color)

# CURVE

times = np.arange(0, 100, 0.2)
values = np.cos(times)*(1+times) + times
coordinates = np.vstack((times, values)).T

# test of basic Curve operations
curve = Curve(coordinates, fullname='original curve', time_range=(0,100))
curve.check()
assert len(curve) == times.size

curve.add_keyframe(-1, 12)
curve.move_keyframe(0, 0, 10)
fig = curve.display(handles=False)

curve.value_transl(-6)
curve.rename('value translation')
curve.display(fig=fig, handles=False)

curve.time_transl(10)
curve.rename('time translation')
curve.display(fig=fig, handles=False)

curve.value_scale(scale=0.5)
curve.rename('value scale')
curve.display(fig=fig, handles=False)

curve.time_scale(scale=0.3)
curve.rename('time scale')
curve.display(fig=fig, handles=False)

curve.get_attribute("value")

curve.check()

values = curve.get_values()
print(f"Shape of curve values: {values.shape}")

if DO_SHOW: fig.show()

# test of curve self checking
wrong_curve1, wrong_curve2 = Curve(), Curve()
wrong_curve1.array[0, 3] = 2.5 # out of api modification -> bypasses checks
wrong_curve2.array[0, 3] = 6 # out of api modification -> bypasses checks
print("Expect two assertion errors...")
try:
    wrong_curve1.check() # should raise AssertionError
except AssertionError as e:
    print(f"As expected, got assertion error: {e}")
try:
    wrong_curve2.check() # should raise AssertionError
except AssertionError as e:
    print(f"As expected, got assertion error: {e}")

# test of derivative on keyframes (using tangent handles)
times = np.zeros((10))
values = np.zeros((10))
co = np.vstack((times, values)).T

left_x = times - np.array([0.1, 0.2, 0.05, 0.2, 0.05, 0.05, 0.3, 0.25, 0.15, 0.15])
right_x = times + np.array([0.1, 0.05, 0.2, 0.2, 0.1, 0.05, 0.3, 0.25, 0.15, 0.05])

left_y = np.array([-0.1, -0.2, -0.05, 0.05, 0.05, -0.15, -0.15, 0.15, 0.35, 0.2])
right_y = np.array([0.1, 0.05, 0.2, -0.05, -0.1, 0.15, 0.15, -0.15, -0.35, 0.25])

# expected derivatives : [ 1, 1, 1,  -0.25, -1, 3, 0.5, -0.6 , -2.33, broken->0]

curve_with_handles = Curve(
    co,
    tangent_left_handle_x=left_x,
    tangent_left_handle_y=left_y,
    tangent_right_handle_x=right_x,
    tangent_right_handle_y=right_y,
    fullname='original curve',
    pointer = FakeCurvePointer(times, values)
)
curve_with_handles.set_keyframe_attribute(9, "handle_left_type", "FREE")
curve_with_handles.set_keyframe_attribute(9, "handle_right_type", "FREE")
derivatives = curve_with_handles.get_keyframe_derivatives()

fig1 = curve_with_handles.display()
print(f"Keyframe derivatives: {list(derivatives)}")

if DO_SHOW: fig1.show()

# test of derivative when we do not have a pointer (using finite schemes)
original_curve = Curve(coordinates, fullname='original curve')

first_derivative = original_curve.first_derivative()
second_derivative = original_curve.second_derivative()

fig2 = first_derivative.display(handles=False)
second_derivative.display(handles=False, fig=fig2)

if DO_SHOW: fig2.show()

# test of apply_warp
def temporal_warp(t,x):
    return 1.5*t, x

def spatial_warp(t,x):
    return t, 1.5*x

def complex_warp(t,x):
    return np.sqrt(abs(x+t))*7, (x/(t+1))*60

original_curve = Curve(coordinates, fullname='original curve')

warped_curve1 = original_curve.apply_spatio_temporal_warp(temporal_warp, in_place=False)
warped_curve1.rename("Temporal warp")

warped_curve2 = original_curve.apply_spatio_temporal_warp(spatial_warp, in_place=False)
warped_curve2.rename("Spatial warp")

warped_curve3 = original_curve.apply_spatio_temporal_warp(complex_warp, in_place=False)
warped_curve3.rename("Complex warp")

fig3 = original_curve.display(handles=False)
warped_curve1.display(fig=fig3, handles=False)
warped_curve2.display(fig=fig3, handles=False)
warped_curve3.display(fig=fig3, handles=False)

if DO_SHOW: fig3.show()

# test of autocropping with handles
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
    tangent_right_handle_y=right_handle_y,
    pointer = FakeCurvePointer(times, values)
)

start, stop = fake_blender_curve.get_auto_crop()

fig = fake_blender_curve.display(style='markers+lines')
fig.add_vline(x=start, annotation_text="Start", annotation_position="top right")
fig.add_vline(x=stop, annotation_text="Stop", annotation_position="top right")

if DO_SHOW: fig.show()

times = np.array(list(range(15)))
values = np.array([0, 0, 6, 3, 0, 2, 4, 2, 0, 1, 3, 1, 0, 0, 0])
handles_left_x = times - 0.25
handles_right_x = times + 0.25
handles_left_y = np.array([0, 0, 6, 6, 0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 0])
handles_right_y = np.array([0, 0, 6, 0, 0, 4, 4, 0, 0, 3, 3, 0, 0, 0, 0])
curve_with_handles = Curve(
    np.vstack((times, values)).T,
    tangent_left_handle_x=handles_left_x,
    tangent_left_handle_y=handles_left_y,
    tangent_right_handle_x=handles_right_x,
    tangent_right_handle_y=handles_right_y,
)
start, stop = curve_with_handles.get_auto_crop()
print(f"Found autocrop of {(start,stop)} when using handles")

# test of autocropping without handles
mag_means = np.zeros((100))
mag_means[30:60] = 12
mag_means[15:18] = 10
mag_means[5] = 6
mag_means[80:85] = 10
mag_means[92] = 6
frame_times = np.arange(0,10,0.1)
curve = Curve(np.vstack((frame_times, mag_means)).T)

for patience in [0, 1, 5]:
    crop = curve.get_auto_crop(use_handles=False, threshold=5./12, padding_in=0, padding_out=0, patience=patience)
    print(f'Cropping found for patience={patience} and threshold=5: {crop}')
crop_other = curve.get_auto_crop(use_handles=False, threshold=0.9, padding_in=0, padding_out=0, patience=0)
print(f'Cropping found for patience=0 and threshold=11: {crop_other}')

curve_no_handles = Curve(
    np.vstack((times, values+np.array([random()*0.3 for _ in range(times.size)]))).T,
)
start, stop = curve_no_handles.get_auto_crop(use_handles=False, padding_out=0, padding_in=0)
print(f"Found autocrop of {(start,stop)} whithout using handles")

# test of curve stitching and blending

curve1 = Curve(coordinates, fullname='first curve')
curve2 = Curve(coordinates, fullname='second curve')
curve2.value_scale(scale=5)

stitched_curve = Curve(coordinates, fullname='stitched curve')
stitched_curve.stitch(curve2)
stitched_curve.value_transl(13*20)

blended_curves : list[Curve] = []
for blend in range(10):
    blended_curve = Curve(coordinates, fullname=f'blended curve : {blend}')
    blended_curve.stitch(curve2, blend=blend)
    blended_curve.value_transl((blend+1)*20)
    blended_curves.append(blended_curve)

fig = curve1.display(handles=False, style="markers+lines")
curve2.display(handles=False, style="markers+lines", fig=fig)
for i,curve in enumerate(blended_curves+[stitched_curve]):
    curve.display(handles=False, style="markers+lines", fig=fig)
if DO_SHOW: fig.show()


## ANIMATION

empty_anim = Animation()
print(f"Empty Animation: {empty_anim}") # expect []

times = np.arange(0, 100, 1) # low resolution
values = np.cos(times)*(1+times) + times
coordinates = np.vstack((times, values)).T

anim = Animation([
    Curve(coordinates, fullname='original curve', pointer=FakeCurvePointer(times, values)),
    Curve(pointer=FakeCurvePointer())
])

anim[0].array # autocomplete: should be blue !

print(f"Animation: {anim}")

resampled_anim = anim.sample(2000)
for param_start, param_stop in zip(["all", "each", 10, [20, 5]],["all", "each", 90, [80, 10]]):
    other = anim.sample(20000, start=param_start, stop=param_stop)
    print(f"start={param_start}, stop={param_stop} -> sampled_anim={other}")


print(f"Sampled animation: {resampled_anim}") # every curve should be of length 2000 !

figure = resampled_anim.display(handles=False) # FakeCurveEvaluator does not handle interpolation -> plateaux
anim.display(handles=False, fig=figure, doShow=DO_SHOW)

derivative = resampled_anim.find("original curve sampled").first_derivative() # test of derivative when we have a pointer
derivative.display(handles=False, style='lines', doShow=DO_SHOW)


addition = empty_anim + anim + resampled_anim
print(f"Addition of all animations: {addition}")

resampled_anim.crop(start=50)
resampled_anim[0].rename("cropped curve")
resampled_anim.display(handles=False, doShow=DO_SHOW)
print("New time range: ", [curve.time_range for curve in resampled_anim])


PATH = 'C:/Users/Marie Bienvenu/stage_m2/afac/'

for curve in anim:
    curve.save(PATH+curve.fullname+'.txt')
    loaded = Curve.load(PATH+curve.fullname+'.txt')
    print(f"Saved curve: {curve}, loaded curve: {loaded}")

anim.save(PATH+'/afac2/')
loaded_anim = Animation.load(PATH+'/afac2/')
print(f"Saved animation: {anim}, loaded animation: {loaded_anim}")


request = anim.find("original curve")
print(f"Curve fetched when asking for 'original curve': {request}")
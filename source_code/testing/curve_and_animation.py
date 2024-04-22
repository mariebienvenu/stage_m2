import numpy as np

from app.Curve import Curve, Color
from app.Animation import Animation

class FakeCurvePointer:

    def __init__(self, times=None, values=None):
        self.times = times
        self.values = values
        # assuming ordered times

    def evaluate(self, time):
        if self.times is None or self.values is None: return None
        for t,v in zip(self.times, self.values):
            if t>=time:
                return v
            
DO_SHOW = False
            
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

# test of derivative when we do not have a pointer
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
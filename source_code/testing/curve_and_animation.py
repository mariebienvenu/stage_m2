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
            
# COLOR
            
for _ in range(20):
    print(Color.next())

# CURVE

times = np.arange(0, 100, 0.2)
values = np.cos(times)*(1+times) + times
coordinates = np.vstack((times, values)).T

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

curve.check()

values = curve.get_values()
print(values.shape)

fig.show()

## ANIMATION

empty_anim = Animation()
print(empty_anim) # expect []

times = np.arange(0, 100, 1) # low resolution
values = np.cos(times)*(1+times) + times
coordinates = np.vstack((times, values)).T

anim = Animation([
    Curve(coordinates, fullname='original curve', pointer=FakeCurvePointer(times, values)),
    Curve(fullname='none', pointer=FakeCurvePointer())
])

anim[0].array # autocomplete: should be blue !

print(anim)

resampled_anim = anim.resample(2000)
resampled_anim[0].rename('resampled curve')

print(resampled_anim) # every curve should be of length 2000 !

figure = resampled_anim.display(handles=False) # FakeCurveEvaluator does not handle interpolation -> plateaux
anim.display(handles=False, fig=figure, doShow=True)

addition = empty_anim + anim + resampled_anim
print(addition)

resampled_anim.crop(start=50)
resampled_anim.display(handles=False, doShow=True)
print("New time range: ", [curve.time_range for curve in resampled_anim])
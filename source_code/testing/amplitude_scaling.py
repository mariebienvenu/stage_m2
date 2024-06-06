
import numpy as np

from app.Curve import Curve
import app.Warp as Warp
import app.maths_utils as m_utils
import app.visualisation as vis


# "multiplicative" warping : y'(i) = y(i)*s(t(i)) avec s un "scaling" linéaire par morceau

## First, let's create some data
np.random.seed(7) #5

t = np.linspace(1, 10, 100)
x1, x2 = np.random.random((100)), np.random.random((100))
x1[[10, 30, 72]] = [6, -5, 3]
x2[[10, 30, 78]] = [4, -8, 3]

## Now the definitions

class AmplitudeWarping(Warp.AbstractWarp):

    def __init__(self, times, multiplicative_factors):
        super(AmplitudeWarping, self).__init__(np.array(times), np.array(multiplicative_factors))

    def __call__(self, t, x):
        return t, np.interp(t, self.input_data, self.output_data)*x
    

def energy(curve:Curve, time_range:tuple[float]):
    time, value = curve.get_times(), curve.get_values()
    indexes = np.array(list(range(time.size)))
    def condition(x):
        if x > time_range[1] : return False
        if x < time_range[0] : return False
        return True
    """condition1 = time<=time_range[1]
    condition2 = time>=time_range[0]
    fulfilled = np.where(condition1 , np.where(condition2), False)"""
    fulfilled = np.vectorize(condition)(time)
    idxs_in_time = indexes[fulfilled]
    x,y = time[idxs_in_time], value[idxs_in_time]
    return m_utils.integrale3(array=y, x=x)[-1]

def multiplicative_factor(curve1:Curve, curve2:Curve, spike_time:float, time_window=0.5):
    energy1 = energy(curve1, (spike_time-time_window, spike_time+time_window))
    energy2 = energy(curve2, (spike_time-time_window, spike_time+time_window))
    return energy2/energy1

## Let's compute

curve1 = Curve(coordinates=np.vstack((t,x1)).T, fullname="reference curve")
curve2 = Curve(coordinates=np.vstack((t,x2)).T, fullname="target curve")

spike_times = [t[10], t[30],t[72]]
window_size = 0.3
multiplicative_factors = [multiplicative_factor(curve1, curve2, spike, window_size) for spike in spike_times]
print(spike_times, multiplicative_factors)

warp = AmplitudeWarping(spike_times, multiplicative_factors)
new_curve = curve1.apply_spatio_temporal_warp(warp, in_place=False)
new_curve.rename("computed curve")

## Now, let's visualize

fig = curve1.display(handles=False, style="lines")
curve2.display(handles=False, style="lines", fig=fig)
new_curve.display(handles=False, style="lines", fig=fig)
fig.show()
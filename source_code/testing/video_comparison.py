
import os
import numpy as np
from plotly.subplots import make_subplots

import app.animation as Animation
import app.curve as Curve
import app.maths_utils as m_utils
import app.visualisation as vis
from app.dynamic_time_warping import DynamicTimeWarping

# Here we define which videos we will compare
data_path = 'C:/Users/Marie Bienvenu/stage_m2/irl_scenes/'
assert os.path.exists(data_path), "Wrong PATH"

subdirectory = '04-08 appareil de Damien'
video_reference = 'P1010236' # original video
video_target = 'P1010236_x2' # timing target

# Loading of the reference curves and computation of some additionnal features
ref_movement = Animation.Animation().load(f'{data_path}/{subdirectory}/{video_reference}/')
ref_frame_times = ref_movement[0].get_times()
additionnal = [curve.first_derivative() for curve in ref_movement]+[curve.second_derivative() for curve in ref_movement]
additionnal_curves_ref = Animation.Animation(
    [curve for curve in additionnal if curve.get_value_range()[1]-curve.get_value_range()[0]>1e-2]
)
additionnal_curves_ref.display(handles=False, style='markers+lines', doShow=True)

# Same, but for the target curves
target_movement = Animation.Animation().load(f'{data_path}/{subdirectory}/{video_target}/')
target_frame_times = target_movement[0].get_times()
additionnal = [curve.first_derivative() for curve in target_movement]+[curve.second_derivative() for curve in target_movement]
additionnal_curves_target = Animation.Animation(
    [curve for curve in additionnal if curve.get_value_range()[1]-curve.get_value_range()[0]>1e-2]
)
additionnal_curves_target.display(handles=False, style='markers+lines', doShow=True)

# Let's take a look at some correlation coefficients - caution: the correlation computation requires the same amount of samples on both curves, so we need to downsample the reference.
subsampled_ref = Animation.Animation([Curve.Curve.from_array(curve.array[::2,...], fullname=curve.fullname+' - subsampled') for curve in ref_movement])
subsampled_additionnal_ref = Animation.Animation([Curve.Curve.from_array(curve.array[1::2,...], fullname=curve.fullname+' - subsampled') for curve in additionnal_curves_ref])
fig0 = make_subplots(rows=1, cols=2, subplot_titles=['Heatmap of basic channels correlations','Heatmap of additionnal channels correlation'])
for i,(r, t) in enumerate(zip([subsampled_ref, subsampled_additionnal_ref], [target_movement, additionnal_curves_target])):
    df = r % t # correlation matrix
    vis.add_heatmap(df, is_correlation=True, max=1, min=-1, fig=fig0, row=1, col=i+1)
    fig0.update_xaxes(title_text=f'Curves of {video_target}', row=1, col=i+1)
fig0.update_layout(title_text=f"Correlation between {video_reference} and {video_target}", yaxis_title=f'Curves of {video_reference}')
fig0.write_html(f'{data_path}/{subdirectory}/{video_reference}_and_{video_target}_correlation.html')
fig0.show()

# Let's pick our curves of interest
chosen_channel = 'First derivative of Velocity Y'
ref = (ref_movement+additionnal_curves_ref).find(chosen_channel)
ref.rename(f'{chosen_channel} - reference')
target = (target_movement+additionnal_curves_target).find(chosen_channel)
target.rename(f'{chosen_channel} - target')

# First visualisation : the curves of interest on separate plots
fig1 = make_subplots(rows=2, cols=1, subplot_titles=["Reference", "Target"])
ref.display(fig=fig1, handles=False, style='lines+markers', col=1, row=1)
target.display(fig=fig1, handles=False, style='lines+markers', col=1, row=2)
fig1.show()

# Let's make them all start at frame 1
ref.time_transl(1-np.min(ref.get_times()))
target.time_transl(1-np.min(target.get_times()))

# And have same mean and std, since we only care about timing here
mean, std = np.mean(ref.get_values()), np.std(ref.get_values())
target.value_transl(mean-np.mean(target.get_values()))
target.value_scale(center=mean, scale=std/np.std(target.get_values()))

# Dynamic Time Warping computation
y1, y2 = ref.get_values(), target.get_values()
x1, x2 = ref.get_times(), target.get_times()
co1, co2  = np.vstack((x1,y1)).T, np.vstack((x2,y2)).T
dtw = DynamicTimeWarping(curve1=Curve.Curve(coordinates=co1), curve2=Curve.Curve(coordinates=co2))
score, pairings = dtw.score, dtw.pairings

# Second visualisation : the curves on the same plot, with pairings as computed by DTW
fig2 = ref.display(handles=False, style='lines')
target.value_transl(2)
target.display(fig=fig2, handles=False, style='lines')
vis.add_pairings(y1, y2+2, pairings, x1, x2, fig=fig2)
fig2.show()

# Third visualisation : the time warping function, compared with some affine approximation
x = [x1[i] for i,j in pairings] 
y = [x2[j] for i,j in pairings]
fig3 = vis.add_curve(y=[min(y), max(y)], x=[min(x), max(x)], name="Affine approximation")
vis.add_curve(fig=fig3, y=y, x=x)
fig3.show()

# Finally : let's "project" one curve onto the other
fig4 = ref.display(handles=False, style='lines')
retimed_x2 = [x[y.index(xi)] for xi in x2]
vis.add_curve(fig=fig4, y=y2, x=retimed_x2, name=f'{target.fullname} - projected')
fig4.show()

target.value_transl(-2)
fig5 = target.display(handles=False, style='lines')
retimed_x1 = [y[x.index(xi)] for xi in x1]
vis.add_curve(fig=fig5, y=y1, x=retimed_x1, name=f'{ref.fullname} - projected')
fig5.show()
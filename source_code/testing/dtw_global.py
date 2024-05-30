
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

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

import plotly
plotly.io.kaleido.scope.mathjax= None # supposedly makes saving figures faster
 
import app.Main as main
import app.visualisation as vis

import importlib
importlib.reload(main)
importlib.reload(main.absIO)
importlib.reload(main.InternalProcess)
importlib.reload(main.InternalProcess.Warp)
importlib.reload(main.InternalProcess.DynamicTimeWarping)
importlib.reload(main.SoftIO)
importlib.reload(main.SoftIO.b_utils)
importlib.reload(main.VideoIO)
importlib.reload(main.VideoIO.Animation)
importlib.reload(main.VideoIO.Animation.Curve)
importlib.reload(vis)

Warp = main.InternalProcess.Warp
DTW = main.InternalProcess.DynamicTimeWarping

Color = main.VideoIO.Animation.Curve.Color
Color.reset()
Color.next()

possible_features = ["Velocity Y", "First derivative of Velocity Y", "Location Y", "First derivative of Location Y", "Second derivative of Location Y"]
chosen_feature = possible_features[-1]
## il se trouve que c'est Location Y qui est la meilleure in fine sur le score, mais Second derivative of Location Y gère mieux le timing des rebonds

directory = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes/bouncing_ball_retiming/"

SHOW = False

## Let's process our case
main.Main.DTW_CONSTRAINTS_LOCAL = 10
main_obj = main.Main(directory, verbose=2)
main_obj.connexions_of_interest[0]["video feature"] = chosen_feature
main_obj.process(force=True)

## A little helper function
def integrale(y, x=None, debug=False):
    y = np.array(y)
    N = y.size
    if x is None: x = np.linspace(1, N, N)
    primitive = np.zeros(N)
    for i in range(1, N):
        primitive[i] = primitive[i-1] + (y[i]+y[i-1])*(x[i]-x[i-1])/2
    return primitive[-1] if debug is False else (primitive[-1], {"Primitive":primitive})

def distL1(x1, y1, x2, y2):
    ## On part du principe que x1 et x2 sont pareils à répétitions près et y1 et y2 aussi (points d'une grille)
    x1, y1, x2, y2 = np.array(x1), np.array(y1), np.array(x2), np.array(y2)
    start_x = x1[0]
    current_x = start_x
    dist = 0
    index_1 = 1
    index_2 = 1
    while index_1<y1.size and index_2<y2.size:
        current_x = min(x1[index_1-1], x2[index_2-1])
        next_x1 , next_x2 = x1[index_1], x2[index_2]
        if next_x1 == current_x:
            index_1 += 1
        elif next_x2 == current_x:
            index_2 += 1
        else:
            assert next_x2==next_x1
            previous_y1, previous_y2 = y1[index_1-1], y2[index_2-1]
            next_y1, next_y2 = y1[index_1], y2[index_2]
            previous_diff = abs(previous_y2 - previous_y1)
            next_diff = abs(next_y2 - next_y1)
            dist += (previous_diff+next_diff)/(2*(next_x2-current_x))
            index_1 += 1
            index_2 += 1
    return dist


x = [-1, 0, 0, 1, 1, 2, 3, 4]
y = [0, 1, 2, 2, 3, 4, 4, 5]
assert integrale(y,x)==14.5

x_prime = [-1, 0, 0, 1, 2, 3, 4, 4, 4]
y_prime = [0, 1, 2, 3, 3, 2, 3, 4, 5]
assert distL1(x, y, x_prime, y_prime)==4.5

## Now, let's get the info we want
dtw:DTW.DynamicTimeWarping = main_obj.internals[0].dtw
best_path = dtw.pairings
constraints = dtw.global_constraints() # This takes a while
distances, paths = dtw.global_constraints_distances, dtw.global_constraints_alternative_paths
paths = paths[1:-1] # first and last are None because we can't remove starting point or final point
bijections = [(
        np.array([dtw.times1[i] for i,j in path]),
        np.array([dtw.times2[j] for i,j in path]),
    ) for path in paths
]
integrales = [integrale(bijection[1], bijection[0]) for bijection in bijections]
integrale_ref = integrale(dtw.bijection[1], dtw.bijection[0])
diffs = np.array([abs(integrale - integrale_ref)/integrale_ref*100 for integrale in integrales])
diffs_ = np.array([distL1(bij_x,bij_y, dtw.bijection[0], dtw.bijection[1])/integrale_ref*100 for bij_x,bij_y in bijections])
## Les deux sont pareils parce que dans notre cas, nos chemins ne se croisent pas ! Ils ne forme qu'une poche de difference puisque le but c'est de prendre au plus court (chemin comme chaîne de Markov, toute solution qui se croise devient égale après...)

## Define the thresholds and filter
CONSTRAINT_THRESHOLD = 3
AREA_THRESHOLD = 0.3
pair_indexes = list(range(1, len(paths)))
is_constrained_enough = [constraints[index]>CONSTRAINT_THRESHOLD for index in pair_indexes]
is_similar_enough = [diffs[index-1]<AREA_THRESHOLD for index in pair_indexes]
is_local_max = [constraints[index]>=max(constraints[index-1], constraints[index+1]) for index in pair_indexes]
kept_indexes = [0] + [index for i,index in enumerate(pair_indexes) if is_constrained_enough[i] and is_similar_enough[i] and is_local_max[i]] + [len(constraints)-1]

discarded_for_constraint_value = [index for i,index in enumerate(pair_indexes) if (not is_constrained_enough[i]) and is_similar_enough[i] and is_local_max[i]]
discarded_for_integration_value = [index for i,index in enumerate(pair_indexes) if (not is_similar_enough[i]) and is_constrained_enough[i] and is_local_max[i]]
discarded_for_local_maxima = [index for i,index in enumerate(pair_indexes) if (not is_local_max[i]) and is_similar_enough[i] and is_constrained_enough[i]]

print(f"Kept indexes: {kept_indexes}")
print(f"Kept pairs: {[dtw.pairings[index] for index in kept_indexes]}")
print(f"Corresponding times: {[(dtw.bijection[0][index],dtw.bijection[1][index]) for index in kept_indexes]}")
print(f"Discarded for being not constrained enough : {discarded_for_constraint_value}")
print(f"Discarded for being too far from ideal path : {discarded_for_integration_value}")
print(f"Discarded for not being a local constraint maxima : {discarded_for_local_maxima}")
warp = Warp.LinearWarp1D(X_in=[dtw.bijection[0][index] for index in kept_indexes], X_out=[dtw.bijection[1][index] for index in kept_indexes])

## And let's start visualizing !

# First, local VS global constraints
fig1 = vis.add_curve(y=dtw.local_constraints(), name="local")
vis.add_curve(y=constraints, name="global", fig=fig1)
fig1.update_layout(
     title="Local VS global constraints on the warping path",
     xaxis_title="Pairs over time",
     yaxis_title="Degree of constraint",
)
if SHOW: fig1.show()

# Next, the alternative paths chosen by the global algo
fig2 = go.Figure()
for i,(path, diff, bijection) in enumerate(zip(paths,diffs, bijections)):
     fig2.add_trace(go.Scatter(
        y=bijection[1],
        x=bijection[0],
        line_color=f'rgba(255,100,0,{diff/max(diffs)})',
        name=f"{i+1}",
    ))
vis.add_curve(
    y=dtw.bijection[1],
    x=dtw.bijection[0],
    color='rgb(100,100,255)',
    fig=fig2,
    name="reference"
)
fig2.update_layout(
    title="Alternative and reference time bijections",
    xaxis_title="Time (frames) - initial",
    yaxis_title="Time (frames) - retook",
)
if SHOW: fig2.show()

# Next, the difference between said paths and the reference one (the optimum)
fig3 = vis.add_curve(y=diffs, x=list(range(1, len(paths))))
fig3.update_layout(
    title="Difference between alternative paths and reference path",
    xaxis_title="Pairs over time",
    yaxis_title="Absolute difference in L1 norm (in %)",
)
if SHOW: fig3.show()

# Next, the warping and the "kept" indexes
fig4 = vis.add_curve(y=dtw.bijection[1], x=dtw.bijection[0], name="DTW matches", style="lines")
vis.add_curve(y=warp.output_data, x=warp.input_data, style="markers", fig=fig4, name="Warp reference points")
x = np.arange(warp.input_data[0], warp.input_data[-1]+1, 1)
vis.add_curve(y=warp(x,None)[0], x=x, style="lines", fig=fig4, name="Warp function")
fig4.update_layout(title="Time bijection", xaxis_title="Time (frames) - initial", yaxis_title="Time (frames) - retook")
if SHOW: fig4.show()

## Let's sum up all this with a recap figure
Color.next() # prettier
kept_color, diff_color, maxim_color = f"rgb{Color.next()}",f"rgb{Color.next()}", f"rgb{Color.next()}"

fig5 = make_subplots(
    rows=2, cols=2, 
    shared_xaxes="rows", 
    subplot_titles=["Global constraints along path", "Integration difference along path", "Alternative and reference paths", "Selected points and final warp"],
    vertical_spacing=0.1, horizontal_spacing=0.1,
)
fig5.update_layout(
    xaxis1_title = "Pairs over time",
    xaxis2_title = "Pairs over time",
    xaxis3_title = "Times (frames) - initial",
    xaxis4_title = "Times (frames) - initial",
    yaxis1_title = "Magnitude of constraint",
    yaxis2_title = "Absolute difference in L1 norm (%)",
    yaxis3_title = "Times (frames) - retook",
    yaxis4_title = "Times (frames) - retook",
)

vis.add_curve(y=constraints, name="Global constraints", row=1, col=1, fig=fig5)

vis.add_curve(y=diffs, x=list(range(1, len(diffs)+1)), name="Integration relative differences", row=1, col=2, fig=fig5)

#r,g,b = Color.next()
#color = f'rgba{(r,g,b,0.3)}'
for i,(index,bijection) in enumerate(zip(pair_indexes, bijections)):
    bij_x, bij_y = bijection
    color = None
    if is_constrained_enough[i] and is_local_max[i] and not is_similar_enough[i]:
        color = diff_color
    elif is_constrained_enough[i] and is_similar_enough[i] and not is_local_max[i]:
        color = maxim_color
    elif index in kept_indexes:
        color = kept_color
    else:
        continue
    vis.add_curve(y=bij_y, x=bij_x, color=color, name=f"Alternative path n°{i+1}", row=2, col=1, fig=fig5, legend=False)

'''for i,(path, diff, bijection) in enumerate(zip(paths,diffs, bijections)):
    bij_x, bij_y = bijection
    vis.add_curve(y=bij_y, x=bij_x, color=color, name=f"Alternative path n°{i+1}", row=2, col=1, fig=fig5, legend=False)'''

color = f"rgb{Color.next()}"
bij_x_ref, bij_y_ref = dtw.bijection
vis.add_curve(y=bij_y_ref, x=bij_x_ref, name="Reference path", color=color, row=2, col=1, fig=fig5)

x = np.arange(warp.input_data[0], warp.input_data[-1]+1, 1)
vis.add_curve(y=bij_y_ref, x=bij_x_ref, color=color, name="Detailed path", style="lines", row=2, col=2, fig=fig5, legend=False)
vis.add_curve(y=warp(x,None)[0], x=x, style="lines", name="Final warp", row=2, col=2, fig=fig5)
vis.add_curve(y=warp.output_data, x=warp.input_data, style="markers", name="Kept points in path", row=2, col=2, fig=fig5)

for index_list, color, name in zip([kept_indexes, discarded_for_integration_value, discarded_for_local_maxima], [kept_color, diff_color, maxim_color], ["Selected", "Discarded - too different of reference path", "Discarded - not an extremum of constraint curve"]):
    for index in index_list:
        i,j = bij_x_ref[index], bij_y_ref[index]
        '''fig5.add_shape(
            type="circle",
            x0=i-2, y0=j-2, x1=i+2, y1=j+2,
            line_color=color,
            row=2, col=1,
            name=name,
            showlegend=(index==index_list[0]),
        )'''
        for colomn in [1,2]:
            fig5.add_vrect(
                x0=index-0.5, x1=index+0.5,
                opacity=0.5,
                line_width=0,
                fillcolor=color,
                row=1, col=colomn,
                name=name,
                showlegend=(index==index_list[0] and colomn==1),
            )

fig5.add_hline(y=CONSTRAINT_THRESHOLD, line_dash="dash", annotation_text=f"Additionnal cost > {CONSTRAINT_THRESHOLD}", opacity=0.7, row=1, col=1)
fig5.add_hline(y=AREA_THRESHOLD,  line_dash="dash", annotation_text=f"Integration difference < {AREA_THRESHOLD}%", opacity=0.7, row=1, col=2)

if SHOW: fig5.show()



## New visualisation : the pairings done by an alternate path compared to original pairings

Color.reset()
Color.next()

def illustrate_alternate_pairings(index:int, color1=Color.next(), color2=Color.next(), fig=None):
    if fig is None : fig=go.Figure()

    alternate_path = paths[index-1]
    forbidden_pair = best_path[index]
    alternate_pairs = [[i,j] for i,j in alternate_path if [i,j] not in best_path]

    x1, y1, x2, y2 = dtw.times1, dtw.values1, dtw.times2, dtw.values2+2

    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=best_path, color="rgb(200, 200, 200)", fig=fig) # optimal path in light grey
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=alternate_pairs, color="green", fig=fig)
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=[forbidden_pair], color="red", fig=fig)

    vis.add_curve(y=y2, x=x2, name="curve1", color=f'rgb{color2}', fig=fig)
    vis.add_curve(y=y1, x=x1, name="curve2",color=f'rgb{color1}', fig=fig)

    return fig

zipped = zip(
    [discarded_for_constraint_value, discarded_for_integration_value, discarded_for_local_maxima, kept_indexes[1:-1]],
    ['discarded for being too cheap compared to ideal pairings', 'discarded for being too far from ideal pairings', 'discarded for not being a local constraint maximum', 'kept'],
    ['too cheap compared to ideal pairings', 'too far from ideal pairings', 'not a local constraint maximum', 'kept'],
)

if False:
    for indexes, name, subdirectory in zipped:
        for index in indexes:
            print(index)
            fig = illustrate_alternate_pairings(index)
            fig.update_layout(title=f'Alternate pairing for index={index} : {name}')
            fig.update_layout(xaxis_title="Time (frames)", yaxis_title="Amplitude (normalized)")
            fig.write_html(directory+f'/out/Feature curve matching simplification/{subdirectory}/{index}.html')
            fig.write_image(directory+f'/out/Feature curve matching simplification/{subdirectory}/{index}.png', height=1080, width=1920) #, scale=2)
            if SHOW: fig.show()


## Let's see what our warped anim would look like with "naive" global constraints, and compare it with refined constraints.

naive_indexes = [0] + [index for i,index in enumerate(pair_indexes) if is_constrained_enough[i]] + [len(constraints)-1]
refined_indexes = kept_indexes

fig = make_subplots(
    rows=2, cols=2, 
    shared_xaxes='all', 
    subplot_titles=["Naive filtering of the matches", "Refined filtering of the matches", "Edited animation using naive warp", "Edited animation using refined warp"],
    vertical_spacing=0.1, horizontal_spacing=0.1,
)

color1, color2 = Color.next(), Color.next()

x1, y1, x2, y2 = dtw.times1+3, dtw.values1, dtw.times2+3, dtw.values2+4
naive_pairings = [e for i,e in enumerate(best_path) if i in naive_indexes]
refined_pairings = [e for i,e in enumerate(best_path) if i in refined_indexes]
vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=best_path, color="rgb(210, 210, 210)", fig=fig, row=1, col=1)
vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=best_path, color="rgb(210, 210, 210)", fig=fig, row=1, col=2)
for col in [1,2]:
    vis.add_curve(y=y2, x=x2, name="curve1", color=f'rgb{color2}', fig=fig, row=1, col=col)
    vis.add_curve(y=y1, x=x1, name="curve2",color=f'rgb{color1}', fig=fig, row=1, col=col)
vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=naive_pairings, color="green", fig=fig, row=1, col=1)
vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=refined_pairings, color="green", fig=fig, row=1, col=2)


time_in, time_out = dtw.bijection
naive_warp = Warp.make_warp(X_in=time_in[naive_indexes]+3, X_out=time_out[naive_indexes]+3)
refined_warp = Warp.make_warp(X_in=time_in[refined_indexes]+3, X_out=time_out[refined_indexes]+3)

edited_naive_animcurve = main_obj.internals[0].make_new_anim(channels=[main_obj.connexions_of_interest[0]['channel']], warps=[naive_warp])
edited_refined_animcurve = main_obj.internals[0].make_new_anim(channels=[main_obj.connexions_of_interest[0]['channel']], warps=[refined_warp])

for new_anim, col in zip([edited_refined_animcurve, edited_naive_animcurve], [2,1]):
    main_obj.new_anims = [new_anim]
    main_obj.to_blender()
    main_obj.blender_scene.from_software(in_place=False)
    blender_anim = main_obj.blender_scene.get_animations()[0]
    blender_anim.display(fig=fig, row=2, col=col)

fig.update_layout(
    xaxis3_title="Time (frames)",
    xaxis4_title="Time (frames)",
    yaxis1_title="Amplitude (arbitrary)",
    yaxis3_title="Amplitude (blender unit)",
)
fig.show()
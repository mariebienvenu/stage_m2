
import os, sys
from tqdm import tqdm

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
from copy import deepcopy

import plotly
plotly.io.kaleido.scope.mathjax= None # supposedly makes saving figures faster
 
import app.main as main
import app.visualisation as vis
import app.dynamic_time_warping as DTW
import app.abstract_io, app.internal_process, app.warping, app.dcc_io, app.blender_utils, app.video_io
import app.animation, app.curve, app.color

Color = app.color.Color
warping = app.warping
Color.reset()

directory = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes/empty/"
SHOW = True
CONSTRAINT_THRESHOLD = 2 #3
AREA_THRESHOLD = 0.1 #0.3
main.Main.DTW_CONSTRAINTS_LOCAL = 10


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

def local_VS_global_constraints(dict):
    fig = vis.add_curve(y=dict["dtw"].local_constraints(), name="local")
    vis.add_curve(y=dict["constraints"], name="global", fig=fig)
    fig.update_layout(
        xaxis_title="Pairs over time",
        yaxis_title="Degree of constraint",
    )
    title = "Local VS global constraints on the warping path"
    return fig, title

def warping_and_kept_indexes(dict):
    fig = vis.add_curve(y=dict["dtw"].bijection[1], x=dict["dtw"].bijection[0], name="DTW matches", style="lines")
    vis.add_curve(y=dict["warp"].output_data, x=dict["warp"].input_data, style="markers", fig=fig, name="Warp reference points")
    x = np.arange(dict["warp"].input_data[0], dict["warp"].input_data[-1]+1, 1)
    vis.add_curve(y=dict["warp"](x,None)[0], x=x, style="lines", fig=fig, name="Warp function")
    fig.update_layout(xaxis_title="Time (frames) - initial", yaxis_title="Time (frames) - retook")
    title="Time bijection"
    return fig, title

def best_path_in_cost_matrix(dict): 
    fig = vis.add_heatmap(pd.DataFrame(dict["dtw"].cost_matrix))
    for index  in dict["kept_indexes"]:
        x,y = dict["bijections"][index-1]
        i,j = dict["bij_y_ref"][index]-dict["bij_y_ref"][0], dict["bij_x_ref"][index]-dict["bij_x_ref"][0]
        color = Color.next()
        vis.add_curve(y=x-x[0], x=y-y[0], name=f'{index}', fig=fig, color=color)
        fig.add_shape(
            type="circle",
            x0=i-2, y0=j-2, x1=i+2, y1=j+2,
            line_color=Color.to_string(color),
            name=f'{index}',
        )
        print(f'Index: {index}, coût:{dict["constraints"][index]}, écart:{dict["diffs"][index-1]}')
    vis.add_curve(y=dict["bij_x_ref"]-dict["bij_x_ref"][0], x=dict["bij_y_ref"]-dict["bij_y_ref"][0], fig=fig)
    title = "Shortest path across cost matrix"
    return fig, title

def cost_along_best_path(dict):
    costs_along_path = [dict["dtw"].cost_matrix[i,j] for i,j in dict["dtw"].pairings]
    fig = vis.add_curve(y=costs_along_path)
    fig.update_layout(xaxis_title="Pairs over time", yaxis_title="Cost in the cost matrix")
    title = "Cost of each pair along the shortest path"
    return fig, title

def score_along_cropping(dict):
    best_scores1 = [np.inf]
    best_scores2 = [np.inf]
    m = len(dict["dtw"].times2)
    n = len(dict["dtw"].times1)
    start1, stop1 = dict["dtw"].curve1.time_range
    start2, stop2 = dict["dtw"].curve2.time_range

    for i in tqdm(range(1,m), desc="Computation of best cropped scores"):
        new_curve = deepcopy(dict["dtw"].curve2)
        cropped = new_curve.crop(start2, start2+i) # in frames ?
        new_dtw = DTW.DynamicTimeWarping(dict["dtw"].curve1, new_curve)
        best_scores2.append(new_dtw.score)

    for i in tqdm(range(1,n), desc="Computation of best cropped scores"):
        new_curve = deepcopy(dict["dtw"].curve1)
        cropped = new_curve.crop(start1, start1+i) # in frames ?
        new_dtw = DTW.DynamicTimeWarping(cropped, dict["dtw"].curve2)
        best_scores1.append(new_dtw.score)

    fig = vis.add_curve(y=best_scores1, x=dict["dtw"].times2, name="Cropping the first curve")
    vis.add_curve(y=best_scores2, x=dict["dtw"].times2, name="Cropping the second curve", fig=fig)
    fig.update_layout(xaxis_title="Time (frames) - retook", yaxis_title="Score of shortest path")
    title="Score along cropping"
    return fig, title

def naive_VS_refined_matches(dict):
    naive_indexes = [0] + [index for i,index in enumerate(dict["pair_indexes"]) if dict["is_constrained_enough"][i]] + [len(dict["constraints"])-1]
    refined_indexes = dict["kept_indexes"]

    fig = make_subplots(
        rows=1, cols=2, 
        shared_xaxes='all', 
        subplot_titles=["Naive filtering of the matches", "Refined filtering of the matches"],
        vertical_spacing=0.1, horizontal_spacing=0.1,
    )

    color1, color2 = Color.next(), Color.next()

    x1, y1, x2, y2 = dict["dtw"].times1, dict["dtw"].values1, dict["dtw"].times2, dict["dtw"].values2+4
    naive_pairings = [e for i,e in enumerate(dict["best_path"]) if i in naive_indexes]
    refined_pairings = [e for i,e in enumerate(dict["best_path"]) if i in refined_indexes]
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=dict["best_path"], color=(210, 210, 210), opacity=1, fig=fig, row=1, col=1)
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=dict["best_path"], color=(210, 210, 210), opacity=1, fig=fig, row=1, col=2)
    for col in [1,2]:
        vis.add_curve(y=y2, x=x2, name="curve1", color=color2, fig=fig, row=1, col=col)
        vis.add_curve(y=y1, x=x1, name="curve2", color=color1, fig=fig, row=1, col=col)
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=naive_pairings, color="green", opacity=1, fig=fig, row=1, col=1)
    vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=refined_pairings, color="green", opacity=1, fig=fig, row=1, col=2)

    fig.update_layout(
        xaxis1_title="Time (frames)",
        xaxis2_title="Time (frames)",
        yaxis_title="Amplitude (arbitrary)",
    )
    title="Naive VS refined matches"
    return fig, title


def big_figure(dict):
    Color.next() # prettier
    kept_color, diff_color, maxim_color = Color.next(), Color.next(), Color.next()

    fig = make_subplots(
        rows=2, cols=2, 
        shared_xaxes="rows", 
        subplot_titles=["Global constraints along path", "Integration difference along path", "Alternative and reference paths", "Selected points and final warp"],
        vertical_spacing=0.1, horizontal_spacing=0.1,
    )
    fig.update_layout(
        xaxis1_title = "Pairs over time",
        xaxis2_title = "Pairs over time",
        xaxis3_title = "Times (frames) - initial",
        xaxis4_title = "Times (frames) - initial",
        yaxis1_title = "Magnitude of constraint",
        yaxis2_title = "Absolute difference in L1 norm (%)",
        yaxis3_title = "Times (frames) - retook",
        yaxis4_title = "Times (frames) - retook",
    )

    vis.add_curve(y=dict["constraints"], name="Global constraints", row=1, col=1, fig=fig)

    vis.add_curve(y=dict["diffs"], x=list(range(1, len(dict["diffs"])+1)), name="Integration relative differences", row=1, col=2, fig=fig)

    for i,(index,bijection) in enumerate(zip(dict["pair_indexes"], dict["bijections"])):
        bij_x, bij_y = bijection
        color = None
        if dict["is_constrained_enough"][i] and dict["is_local_max"][i] and not dict["is_similar_enough"][i]:
            color = diff_color
        elif dict["is_constrained_enough"][i] and dict["is_similar_enough"][i] and not dict["is_local_max"][i]:
            color = maxim_color
        elif index in dict["kept_indexes"]:
            color = kept_color
        else:
            continue
        vis.add_curve(y=bij_y, x=bij_x, color=color, name=f"Alternative path n°{i+1}", row=2, col=1, fig=fig, legend=False)

    color = Color.next()
    bij_x_ref, bij_y_ref = dict["dtw"].bijection
    vis.add_curve(y=bij_y_ref, x=bij_x_ref, name="Reference path", color=color, row=2, col=1, fig=fig)

    x = np.arange(dict["warp"].input_data[0], dict["warp"].input_data[-1]+1, 1)
    vis.add_curve(y=bij_y_ref, x=bij_x_ref, color=color, name="Detailed path", style="lines", row=2, col=2, fig=fig, legend=False)
    vis.add_curve(y=dict["warp"](x,None)[0], x=x, style="lines", name="Final warp", row=2, col=2, fig=fig)
    vis.add_curve(y=dict["warp"].output_data, x=dict["warp"].input_data, style="markers", name="Kept points in path", row=2, col=2, fig=fig)

    for index_list, color, name in zip([dict["kept_indexes"], dict["discarded_for_integration_value"], dict["discarded_for_local_maxima"]], [kept_color, diff_color, maxim_color], ["Selected", "Discarded - too different of reference path", "Discarded - not an extremum of constraint curve"]):
        for index in index_list:
            i,j = bij_x_ref[index], bij_y_ref[index]
            for colomn in [1,2]:
                fig.add_vrect(
                    x0=index-0.5, x1=index+0.5,
                    opacity=0.5,
                    line_width=0,
                    fillcolor=f'rgb{color}',
                    row=1, col=colomn,
                    name=name,
                    showlegend=(index==index_list[0] and colomn==1),
                )

    fig.add_hline(y=CONSTRAINT_THRESHOLD, line_dash="dash", annotation_text=f"Additionnal cost > {CONSTRAINT_THRESHOLD}", opacity=0.7, row=1, col=1)
    fig.add_hline(y=AREA_THRESHOLD,  line_dash="dash", annotation_text=f"Integration difference < {AREA_THRESHOLD}%", opacity=0.7, row=1, col=2)

    title = "Warp refining"
    return fig, title


def distances(dict):
    fig = vis.add_heatmap(pd.DataFrame(dict["distances"]))
    vis.add_curve(y=dict["bij_x_ref"]-dict["bij_x_ref"][0], x=dict["bij_y_ref"]-dict["bij_y_ref"][0], fig=fig)
    title = "Path across distance matrix"
    return fig, title


def make_diagrams(dict):
    makers = [local_VS_global_constraints, warping_and_kept_indexes, big_figure, naive_VS_refined_matches, best_path_in_cost_matrix, cost_along_best_path, score_along_cropping, distances]
    titles : list[str] = []
    figures : list[go.Figure] = []
    for f in makers:
        title, fig = f(dict)
        figures.append(fig)
        titles.append(title)
    return figures,titles


def experiment(ref:str, target:str):
    exp_directory = directory+f'/{ref}_VS_{target}/'
    main_obj = main.Main(directory, no_blender=True, verbose=2)
    main_obj.config["video reference filename"] = ref
    main_obj.config["video target filename"] = target
    main_obj.process(force=True)
    if not os.path.exists(exp_directory): os.mkdir(exp_directory)
    main_obj.draw_diagrams(directory=exp_directory)

    dtw:DTW.DynamicTimeWarping = main_obj.internals[0].dtw
    best_path = dtw.pairings
    bij_x_ref, bij_y_ref = dtw.bijection
    constraints = dtw.global_constraints() # This takes a while
    distances, paths = dtw.global_constraints_distances, dtw.global_constraints_alternative_paths
    paths = paths[1:-1] # first and last are None because we can't remove starting point or final point
    bijections = [(
            np.array([dtw.times1[i] for i,j in path]),
            np.array([dtw.times2[j] for i,j in path]),
        ) for path in paths
    ]
    n = dtw.times1[-1] - dtw.times1[0]
    m = dtw.times2[-1] - dtw.times2[0]
    integrales = [integrale(bijection[1], bijection[0]) for bijection in bijections]
    integrale_ref = integrale(dtw.bijection[1], dtw.bijection[0])
    diffs = np.array([abs(integrale - integrale_ref)/(n*m)*100 for integrale in integrales])

    pair_indexes = list(range(1, len(paths)))
    is_constrained_enough = [constraints[index]>CONSTRAINT_THRESHOLD for index in pair_indexes]
    is_similar_enough = [diffs[index-1]<AREA_THRESHOLD for index in pair_indexes]
    is_local_max = [constraints[index]>=max(constraints[index-1], constraints[index+1]) for index in pair_indexes]
    kept_indexes = [0] + [index for i,index in enumerate(pair_indexes) if is_constrained_enough[i] and is_similar_enough[i] and is_local_max[i]] + [len(constraints)-1]

    discarded_for_constraint_value = [index for i,index in enumerate(pair_indexes) if (not is_constrained_enough[i]) and is_similar_enough[i] and is_local_max[i]]
    discarded_for_integration_value = [index for i,index in enumerate(pair_indexes) if (not is_similar_enough[i]) and is_constrained_enough[i] and is_local_max[i]]
    discarded_for_local_maxima = [index for i,index in enumerate(pair_indexes) if (not is_local_max[i]) and is_similar_enough[i] and is_constrained_enough[i]]

    warp = warping.LinearWarp1D(X_in=[dtw.bijection[0][index] for index in kept_indexes], X_out=[dtw.bijection[1][index] for index in kept_indexes])

    file = open(exp_directory+"info.txt", 'w')
    file.writelines(
        [
            f"Kept indexes: {kept_indexes}",
            f"Kept pairs: {[dtw.pairings[index] for index in kept_indexes]}",
            f"Corresponding times: {[(dtw.bijection[0][index],dtw.bijection[1][index]) for index in kept_indexes]}",
            f"Discarded for being not constrained enough : {discarded_for_constraint_value}",
            f"Discarded for being too far from ideal path : {discarded_for_integration_value}",
            f"Discarded for not being a local constraint maxima : {discarded_for_local_maxima}",
            f"Best score: {dtw.score}"
        ]
    )
    file.close()

    dictionary = {
        "dtw":dtw,
        "best_path":best_path,
        "constraints":constraints,
        "bijections":bijections,
        "diffs":diffs,
        "pair_indexes":pair_indexes,
        "is_constrained_enough":is_constrained_enough,
        "is_similar_enough":is_similar_enough,
        "is_local_max":is_local_max,
        "kept_indexes":kept_indexes,
        "discarded_for_constraint_value":discarded_for_constraint_value,
        "discarded_for_integration_value":discarded_for_integration_value,
        "discarded_for_local_maxima":discarded_for_local_maxima,
        "warp":warp,
        "bij_y_ref":bij_y_ref,
        "bij_x_ref":bij_x_ref,
        "distances":distances,
    }
    figures, titles = make_diagrams(dictionary)
    figures : list[go.Figure]
    titles : list[str] 

    for figure,title in zip(figures, titles):
        figure.update_layout(title=title)
        filetitle = title.replace('"','')
        figure.write_html(f'{exp_directory}/{filetitle}.html')
        if SHOW: figure.show()





filenames = ["P1010258", "P1010259", "P1010260", "P1010261", "P1010262", "P1010263"]

ref = filenames[3]
for target_idx in [0, 1, 2, 4, 5]:
    target = filenames[target_idx]
    experiment(ref, target)
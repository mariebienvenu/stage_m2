
from copy import deepcopy
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from app.animation import Animation
import app.warping as W
from app.dynamic_time_warping import DynamicTimeWarping

import app.image_gradient as im_grad
from app.curve import Curve
from app.color import Color
import app.visualisation as vis


class InternalProcess:

    COST_THRESHOLD = 7
    CONSTRAINT_THRESHOLD = 2
    AREA_THRESHOLD = 0.1
    LOCAL_WINDOW_SIZE = 10


    def __init__(self, vanim_ref:Animation, vanim_target:Animation, banim:Animation, verbose=0):
        self.vanim1 = vanim_ref
        self.vanim2 = vanim_target
        self.banim1 = banim
        self.verbose = verbose # TODO make use of this more


    def process(self, feature:str, channels:list[str], only_temporal=True, detect_issue=True, blend_if_issue=10, filter_indexes=True, warp_interpolation="linear", normalize="True"):
        if not only_temporal: raise NotImplementedError
        self.select(feature)
        if normalize: self.normalize_curves()
        self.do_dtw()
        self.issue_detected = False
        if detect_issue:
            self.issue_detected = self.detect_number_issues()
            if self.issue_detected:
                self.operator = self.process_number_issues(blend=blend_if_issue)
                self.new_curve1 = self.operator(self.curve1)
                if normalize: self.normalize_curves()
                self.do_dtw(redo=True)
                still_issue = self.detect_number_issues()
                #if still_issue: raise RecursionError(f"Internal Process failed to solve outlier issues. \n\t Outliers:{self.outliers} (expected [])")
                if still_issue and self.verbose>0: print(f"Internal Process failed to solve outlier issues. \n\t Outliers:{self.outliers} (expected [])")
        self.make_warp(filter=filter_indexes, interpolation=warp_interpolation)
        self.make_new_anim(channels, use_operator=self.issue_detected)
        return self.banim2


    def select(self, feature:str):
        self.feature = feature
        self.curve1 = self.vanim1.find(feature)
        self.curve2 = self.vanim2.find(feature)


    def normalize_curves(self):
        self.curve1.normalize()
        self.curve2.normalize()
        try:
            self.new_curve1.normalize()
        except AttributeError:
            return


    def do_dtw(self, redo=False):
        self.former_dtw = getattr(self, "dtw", None)
        self.former_dtw_constraints = getattr(self, "dtw_constraints", None)
        self.dtw = DynamicTimeWarping(self.curve1 if not redo else self.new_curve1, self.curve2) # performs the algorithm computation at init time
        self.dtw_constraints = self.dtw.global_constraints()


    def detect_number_issues(self):
        # TODO find a better name for the function, beurk
        self.former_outliers = getattr(self, "outliers", None)
        self.outliers = self.dtw.detect_limitation(InternalProcess.COST_THRESHOLD, InternalProcess.CONSTRAINT_THRESHOLD, InternalProcess.AREA_THRESHOLD, InternalProcess.LOCAL_WINDOW_SIZE)
        if self.verbose>0: print(f"Internal Process : found {len(self.outliers)} outliers in shortest path.")
        return len(self.outliers)!=0


    def process_number_issues(self, blend=False):
        # TODO find a better name for the function, beurk
        gradient = im_grad.image_gradient(self.dtw.cost_matrix)
        pairs = self.dtw.pairings
        outliers_grad = [gradient[:, pairs[index][0], pairs[index][1]] for index in self.outliers]
        needed_action = ["add" if abs(dx)>abs(dy) else "delete" for dx,dy in outliers_grad]
        inliers, outliers = self.dtw.filtered_indexes()[1:-1], self.outliers
        inliers_time_in_ref, outliers_time_in_ref = self.dtw.bijection[0][inliers], self.dtw.bijection[0][outliers] # need to compute this here because dtw might change later on, and then calling operator would no longer have sense
        start, stop = 0, len(self.dtw.pairings)-1  
        m, k = len(inliers), len(self.outliers)
        def operator(input_curve:Curve):
            if self.verbose>0: print(f"InternalProcess - Operator : called on {input_curve} - expected action: {needed_action} {k} impulse(s)")
            if all([action == "delete" for action in needed_action]):
                times_in_ref = np.sort(np.concatenate((outliers_time_in_ref, inliers_time_in_ref))) # len = k+m
                time_for_crop = (times_in_ref[m-1]+times_in_ref[m])/2 # will allow the deletion of the k outliers
                output_curve = deepcopy(input_curve)
                output_curve.crop(stop=time_for_crop)
            if all([action == "add" for action in needed_action]):
                times_in_ref = inliers_time_in_ref # len = m
                time_start, time_stop = self.dtw.bijection[0][[start, stop]]
                delta = blend/2 if not(blend is False or blend is None) else 0
                box_start = (time_start, times_in_ref[-2]+delta)
                box_end = (times_in_ref[-1]-delta, time_stop)
                box_to_duplicate = (times_in_ref[-2]-delta, times_in_ref[-1]+delta)
                output_curve, motif, ending = deepcopy(input_curve), deepcopy(input_curve), deepcopy(input_curve)
                for c, box in zip([output_curve, motif, ending],  [box_start, box_to_duplicate, box_end]):
                    c.crop(box[0], box[1])
                motif_strict_size = box_to_duplicate[1] - box_to_duplicate[0] - 2*delta
                for counter in range(k+1): # motif should be added once per outlier
                    self_end_time = box_start[1]+counter*motif_strict_size
                    curve_start_time = box_to_duplicate[0]
                    if self.verbose>0: print(f"InternalProcess - Operator : computing motif stitch n°{counter+1}/{k+1} with end_self={self_end_time} and start_other={curve_start_time}s")
                    output_curve.stitch(deepcopy(motif), blend=blend, self_end_time=self_end_time, curve_start_time=curve_start_time, verbose=self.verbose-1)
                self_end_time = box_start[1]+(k+1)*motif_strict_size
                curve_start_time = box_end[0]
                if self.verbose>0: print(f"InternalProcess - Operator : computing final ending stitch with end={self_end_time} and start_other={curve_start_time}s")
                output_curve.stitch(ending, blend=blend, self_end_time=self_end_time, curve_start_time=curve_start_time, verbose=self.verbose-1)
            return output_curve
        return operator # can be used for both the reference feature curve, and the reference animation ! yay


    def make_warp(self, filter=True, interpolation="linear"):
        time_in, time_out = self.dtw.bijection
        self.kept_indexes = self.dtw.filtered_indexes() if filter else list(range(len(self.dtw.pairings)))
        if len(self.kept_indexes)<=2: 
            if self.verbose>0: print(f"Warp simplification did not work ; no index was selected. Reverting to dense warp computation.")
            self.kept_indexes = list(range(len(self.dtw.pairings)))
        self.warp = W.make_warp(dimension=1, interpolation=interpolation, X_in=time_in[self.kept_indexes], X_out=time_out[self.kept_indexes])


    def make_new_anim(self, channels:list[str], use_operator=False):
        self.banim2 = Animation()
        if use_operator: self.new_banim1 = Animation()
        for curve in self.banim1:
            if curve.fullname in channels:
                temp_curve = self.operator(curve) if use_operator else curve    
                if use_operator: self.new_banim1.append(temp_curve)
                self.banim2.append(temp_curve.apply_spatio_temporal_warp(self.warp, in_place=False))
            else:
                if use_operator: self.new_banim1.append(curve)
                self.banim2.append(curve)


    def make_diagrams(self, number_issues=True):

        inlier_color, outlier_color, path_color = Color.next(), Color.next(), Color.next()
        ref_color, target_color = Color.next(), Color.next()

        def aux_enriched_map(dtw:DynamicTimeWarping, pairs:list[list[int]], inliers:list[int], outliers:list[int]):
            fig = dtw.make_map(path_color=path_color)
            circle_indexes = inliers+outliers
            circle_colors = [inlier_color]*len(inliers) + [outlier_color]*len(outliers)
            for index, color in zip(circle_indexes, circle_colors):
                x,y = pairs[index][1]-pairs[0][1], pairs[index][0]-pairs[0][0]
                vis.add_circle(center=(x,y), color=color, fig=fig)
            return fig
        
        def aux_matches_filter(dtw:DynamicTimeWarping, pairs:list[list[int]], naive_indexes:list[int], filtered_indexes:list[int], vertical_offset=4):
            fig = make_subplots(
                rows=1, cols=3, 
                shared_xaxes='all', 
                subplot_titles=["No filter", "Basic filter (1 criteria)", "Refined filter  (several criteria)"],
                vertical_spacing=0.1, horizontal_spacing=0.1,
            )

            x1, y1, x2, y2 = dtw.times1, dtw.values1, dtw.times2, dtw.values2+vertical_offset
            all_pairings = pairs
            naive_pairings = [e for i,e in enumerate(pairs) if i in naive_indexes]
            refined_pairings = [e for i,e in enumerate(pairs) if i in filtered_indexes]
            for col in [1,2, 3]:
                vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=all_pairings, color=(210, 210, 210), opacity=1, fig=fig, row=1, col=col)
                vis.add_curve(y=y2, x=x2, name="curve1", color=target_color, fig=fig, row=1, col=col)
                vis.add_curve(y=y1, x=x1, name="curve2", color=ref_color, fig=fig, row=1, col=col)
            vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=all_pairings, color="green", opacity=1, fig=fig, row=1, col=1)
            vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=naive_pairings, color="green", opacity=1, fig=fig, row=1, col=2)
            vis.add_pairings(y2=y2, x2=x2, y1=y1, x1=x1, pairs=refined_pairings, color="green", opacity=1, fig=fig, row=1, col=3)

            fig.update_layout(
                xaxis1_title="Time (frames)",
                xaxis2_title="Time (frames)",
                xaxis3_title="Time (frames)",
                yaxis_title="Amplitude (arbitrary)",
            )
            return fig
        
        def aux_along_path(dtw:DynamicTimeWarping, constraints:np.ndarray, inliers:list[int], outliers:list[int], rectangle_width=1):
            fig = go.Figure()
            rect_indexes = inliers+outliers
            rect_colors = [inlier_color]*len(inliers) + [outlier_color]*len(outliers)
            for index, color in zip(rect_indexes, rect_colors):
                fig.add_vrect(x0=index-rectangle_width/2, x1=index+rectangle_width/2, opacity=0.8, line_width=0, fillcolor=Color.to_string(color))
            vis.add_curve(y=constraints, name="contribution as saved cost", fig=fig)
            vis.add_curve(y=dtw.cost_along_shortest_path, name="instant cost", fig=fig)
            return fig
            
        dtw = self.dtw
        constraints = self.dtw_constraints
        pairs = dtw.pairings
        inliers = self.kept_indexes
        outliers = getattr(self, "outliers", [])
        naive_indexes = [0] + [index for index in range(1, len(pairs)-1) if constraints[index]>InternalProcess.CONSTRAINT_THRESHOLD] + [len(pairs)-1]
        
        fig1 = aux_enriched_map(dtw, pairs, inliers, outliers)
        title1 = "Cost matrix with shortest path and selected nodes"

        fig2 = aux_matches_filter(dtw, pairs, naive_indexes, inliers)
        title2 = "Matches with no or simple or refined selection"

        fig3 = aux_along_path(dtw, constraints, inliers, outliers, 2)
        title3 = "Cost and saved cost along shortest path"

        figures = [fig1, fig2, fig3]
        titles = [title1, title2, title3]

        if not number_issues : return figures, titles

        former_dtw:DynamicTimeWarping = self.former_dtw
        former_constraints = self.former_dtw_constraints
        former_pairs = former_dtw.pairings
        former_outliers = self.former_outliers
        former_inliers = former_dtw.filtered_indexes()
        former_naive_indexes = [0] + [index for index in range(1, len(former_pairs)-1) if former_constraints[index]>InternalProcess.CONSTRAINT_THRESHOLD] + [len(former_pairs)-1]

        fig4 = aux_enriched_map(former_dtw, former_pairs, former_inliers, former_outliers)
        title4 = "Cost matrix with shortest path and selected nodes - before"

        fig5 = aux_matches_filter(former_dtw, former_pairs, former_naive_indexes, former_inliers)
        title5 = "Matches with no or simple or refined selection - before"

        fig6 = make_subplots(rows=2, cols=2, shared_xaxes=True, subplot_titles=["Before", "After"])
        self.curve1.display(handles=False, style="lines+markers", color=ref_color, fig=fig6, row=1, col=1)
        self.new_curve1.display(handles=False, style="lines+markers", color=ref_color, fig=fig6, row=2, col=1)
        title6 = "Reference feature curve - before and after processing"
        
        fig7, title7 = go.Figure(), ""
        if len(self.banim1)>0:
            fig7 = make_subplots(rows=2, cols=2, shared_xaxes=True, subplot_titles=["Before", "After"])
            self.banim1.display(fig=fig7, row=1, col=1)
            self.new_banim1.display(fig=fig7, row=2, col=1)
            title7 = "Original animation curve - before and after processing"

        fig8 = aux_along_path(former_dtw, former_constraints, former_inliers, former_outliers, 2)
        title8 = "Cost and saved cost along shortest path - before"

        figures = [fig6, fig7, fig8, fig4, fig5, fig1, fig2, fig3]
        titles = [title6, title7, title8, title4, title5, title1+" - after", title2+" - after", title3+" - after"]

        return figures, titles
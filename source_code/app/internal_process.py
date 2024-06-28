
from typing import List # TODO should be removed, useless ; and legacy class too
from copy import deepcopy
import numpy as np

from app.animation import Animation
import app.warping as W
from app.dynamic_time_warping import DynamicTimeWarping

import app.image_gradient as im_grad
from app.curve import Curve

class InternalProcess: # TODO: just did a pretty big refactory, should write a dedicated test file 


    COST_THRESHOLD = 2
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
        issue = False
        if detect_issue:
            issue = self.detect_number_issues()
            if issue:
                self.operator = self.process_number_issues(blend=blend_if_issue)
                self.new_curve1 = self.operator(self.curve1)
                if normalize: self.normalize_curves()
                self.do_dtw(redo=True)
                still_issue = self.detect_number_issues()
                if still_issue: raise RecursionError(f"Internal Process failed to solve outlier issues. \n\t Outliers:{self.outliers} (expected [])")
        self.make_warp(filter=filter_indexes, interpolation=warp_interpolation)
        self.make_new_anim(channels, use_operator=issue)
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
        self.dtw = DynamicTimeWarping(self.curve1 if not redo else self.new_curve1, self.curve2) # performs the algorithm computation at init time
        self.dtw_constraints = self.dtw.global_constraints()


    def detect_number_issues(self):
        # TODO find a better name for the function, beurk
        self.outliers = self.dtw.detect_limitation(InternalProcess.COST_THRESHOLD, InternalProcess.COST_THRESHOLD, InternalProcess.AREA_THRESHOLD, InternalProcess.LOCAL_WINDOW_SIZE)
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
        for curve in self.banim1:
            if curve.fullname in channels:
                temp_curve = self.operator(curve) if use_operator else curve                    
                self.banim2.append(temp_curve.apply_spatio_temporal_warp(self.warp, in_place=False))
            else:
                self.banim2.append(curve)



class LegacyInternalProcess:

    def __init__(self, vanim_ref:Animation, vanim_target:Animation, banim:Animation):
        self.vanim1 = vanim_ref
        self.vanim2 = vanim_target
        self.banim1 = banim
    

    def make_warp(self, feature:str=None, only_temporal=True, interpolation="linear", verbose=0):
        self.feature = feature
        curve1 = self.vanim1.find(feature)
        curve2 = self.vanim2.find(feature)
        if not only_temporal: raise NotImplementedError
        curve1.normalize()
        curve2.normalize()
        self.dtw = DynamicTimeWarping(curve1, curve2) # performs the DTW algo at init time
        time_in, time_out = self.dtw.bijection
        warp = W.make_warp(dimension=1, interpolation=interpolation, X_in=time_in, X_out=time_out)
        return warp
    

    def make_simplified_warp(self, feature:str=None, only_temporal=True, constraint_threshold=1.5, local_scale=10, interpolation="linear", verbose=0):
        self.feature = feature
        curve1 = self.vanim1.find(feature)
        curve2 = self.vanim2.find(feature)
        if not only_temporal: raise NotImplementedError
        curve1.normalize()
        curve2.normalize()
        self.dtw = DynamicTimeWarping(curve1, curve2) # performs the DTW algo at init time
        time_in, time_out = self.dtw.bijection
        self.dtw_constraints = self.dtw.local_constraints(local_scale) if local_scale>0 else self.dtw.global_constraints() #measure_dtw_constraint(local=True, window_size=local_scale)
        self.kept_indexes = [0]
        zipped = zip(self.dtw_constraints[:-2], self.dtw_constraints[1:-1], self.dtw_constraints[2:])
        for i,(left,constraint,right) in enumerate(zipped):
            if constraint > max(constraint_threshold, left, right):
                self.kept_indexes.append(i+1)
        self.kept_indexes.append(self.dtw_constraints.size-1)
        if len(self.kept_indexes)==2:
            if verbose>0: print(f"Warp simplification did not work ; no index has a certainty better than {constraint_threshold}. Reverting to classic warp computation.")
            return self.make_warp(feature, only_temporal, interpolation, verbose=verbose)
        warp = W.make_warp(dimension=1, interpolation=interpolation, X_in=time_in[self.kept_indexes], X_out=time_out[self.kept_indexes])
        return warp
        
    
    def make_new_anim(self, channels:List[str]=None, warps:List[W.AbstractWarp]=None):
        banim2 = Animation()
        for curve in self.banim1:
            if curve.fullname in channels:
                warp = warps[channels.index(curve.fullname)]
                banim2.append(curve.apply_spatio_temporal_warp(warp, in_place=False))
            else:
                banim2.append(curve)
        return banim2


    def measure_dtw_constraint(self, local=True, window_size=10): # TODO - remove, useless
        if local: return self.dtw.local_constraints(window_size)
        else : return self.dtw.global_constraints()
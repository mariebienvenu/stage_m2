from typing import List
import numpy as np

import app.Animation as Animation
import app.Warp as Warp
import app.DynamicTimeWarping as DynamicTimeWarping

class InternalProcess:

    def __init__(self, vanim_ref:Animation.Animation, vanim_target:Animation.Animation, banim:Animation.Animation):
        self.vanim1 = vanim_ref
        self.vanim2 = vanim_target
        self.banim1 = banim

    
    def make_warp(self, feature:str=None, only_temporal=True, verbose=0):
        self.feature = feature
        curve1 = self.vanim1.find(feature)
        curve2 = self.vanim2.find(feature)
        if not only_temporal: raise NotImplementedError
        curve1.normalize()
        curve2.normalize()
        self.dtw = DynamicTimeWarping.DynamicTimeWarping(curve1, curve2) # performs the DTW algo at init time
        time_in, time_out = self.dtw.bijection
        warp = Warp.LinearWarp1D(time_in, time_out)
        return warp
    

    def make_simplified_warp(self, feature:str=None, only_temporal=True, uncertainty_threshold=1.5, local_scale=10, verbose=0):
        self.feature = feature
        curve1 = self.vanim1.find(feature)
        curve2 = self.vanim2.find(feature)
        if not only_temporal: raise NotImplementedError
        curve1.normalize()
        curve2.normalize()
        self.dtw = DynamicTimeWarping.DynamicTimeWarping(curve1, curve2) # performs the DTW algo at init time
        time_in, time_out = self.dtw.bijection
        self.dtw_constraints = self.measure_dtw_local_constraint(local_scale)
        self.kept_indexes = [0]
        for i,constraint in enumerate(self.dtw_constraints):
            if constraint > uncertainty_threshold:
                self.kept_indexes.append(i)
        self.kept_indexes.append(self.dtw_constraints.size-1)
        if len(self.kept_indexes)==2:
            if verbose>0: print(f"Warp simplification did not work ; no index has a certainty better than {uncertainty_threshold}. Reverting to classic warp computation.")
            return self.make_warp(feature, only_temporal, verbose)
        warp = Warp.LinearWarp1D(time_in[self.kept_indexes], time_out[self.kept_indexes])
        return warp
        
    
    def make_new_anim(self, channels:List[str]=None, warps:List[Warp.AbstractWarp]=None):
        banim2 = Animation.Animation()
        for curve in self.banim1:
            if curve.fullname in channels:
                warp = warps[channels.index(curve.fullname)]
                banim2.append(curve.apply_spatio_temporal_warp(warp, in_place=False))
            else:
                banim2.append(curve)
        return banim2


    def measure_dtw_local_constraint(self, window_size=10):
        range_x = int(self.dtw.curve1.time_range[1]-self.dtw.curve1.time_range[0])
        range_y = int(self.dtw.curve2.time_range[1]-self.dtw.curve2.time_range[0])
        N = self.dtw.bijection[0].size
        local_constraints = np.zeros((N))
        for i in range(1, N-1):
            ix,iy = np.array(self.dtw.pairings)[::-1][i] ## TODO make pairings an int array from start ? aller voir vis.add_pairings
            center_cost = self.dtw.cost_matrix[ix, iy] # best cost (globally)
            w_size = min(window_size, ix, iy, range_x-ix, range_y-iy)
            upper_costs = self.dtw.cost_matrix[ix+1:ix+w_size, iy] + self.dtw.cost_matrix[ix, iy+1:iy+w_size]
            lower_costs = self.dtw.cost_matrix[ix-w_size:ix, iy] + self.dtw.cost_matrix[ix, iy-w_size:iy]
            alternative_costs = np.concatenate((upper_costs, lower_costs))
            minimal_additionnal_cost = np.min(alternative_costs) - center_cost if alternative_costs.size>0 else 0
            local_constraints[i] = max(0,minimal_additionnal_cost)
        return local_constraints
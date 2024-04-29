from typing import List
import numpy as np

import app.Animation as Animation
import app.Warp as Warp
import app.maths_utils as m_utils

class InternalProcess:

    def __init__(self, vanim_ref:Animation.Animation, vanim_target:Animation.Animation, banim:Animation.Animation):
        self.vanim1 = vanim_ref
        self.vanim2 = vanim_target
        self.banim1 = banim

    
    def make_warp(self, feature:str=None, only_temporal=True, verbose=0):
        self.feature = feature
        curve1 = self.vanim1.find(feature)
        curve2 = self.vanim2.find(feature)
        if only_temporal:
            curve1.normalize()
            curve2.normalize()
            self.warp_value_ref = curve1.get_values()
            self.warp_value_target = curve2.get_values()
            self.warp_time_ref, self.warp_time_target = curve1.get_times(), curve2.get_times()
            self.dtw_score, self.dtw_pairings, self.dtw_array, self.dtw_costmatrix = m_utils.dynamic_time_warping(self.warp_value_ref, self.warp_value_target, debug=True)
            if verbose>0: print(f"DTW achieved a score of {self.dtw_score}.")
            self.x_in = [curve1.get_times()[i] for i,j in self.dtw_pairings[::-1]] 
            self.x_out = [curve2.get_times()[j] for i,j in self.dtw_pairings[::-1]]
            warp = Warp.LinearWarp1D(self.x_in, self.x_out)
            return warp
        raise NotImplementedError
    

    def make_simplified_warp(self, feature:str=None, only_temporal=True, uncertainty_threshold=1.5, local_scale=10, verbose=0):
        self.feature = feature
        curve1 = self.vanim1.find(feature)
        curve2 = self.vanim2.find(feature)
        if not only_temporal: raise NotImplementedError
        curve1.normalize()
        curve2.normalize()
        self.warp_value_ref = curve1.get_values()
        self.warp_value_target = curve2.get_values()
        self.warp_time_ref, self.warp_time_target = curve1.get_times(), curve2.get_times()
        self.dtw_score, self.dtw_pairings, self.dtw_array, self.dtw_costmatrix = m_utils.dynamic_time_warping(self.warp_value_ref, self.warp_value_target, debug=True)
        if verbose>0: print(f"DTW achieved a score of {self.dtw_score}.")
        self.x_in = [self.warp_time_ref[i] for i,j in self.dtw_pairings[::-1]] 
        self.x_out = [self.warp_time_target[j] for i,j in self.dtw_pairings[::-1]]
        self.dtw_constraints = self.measure_dtw_local_constraint(local_scale)
        self.kept_indexes = [0]
        for i,constraint in enumerate(self.dtw_constraints):
            if constraint > uncertainty_threshold:
                self.kept_indexes.append(i)
        self.kept_indexes.append(self.dtw_constraints.size-1)
        if len(self.kept_indexes)==2:
            if verbose>0: print(f"Warp simplification did not work ; no index has a certainty better than {uncertainty_threshold}. Reverting to classic warp computation.")
            return self.make_warp(feature, only_temporal, verbose)
        self.x_in = [self.warp_time_ref[i] for index, (i,j) in enumerate(self.dtw_pairings[::-1]) if index in self.kept_indexes]
        self.x_out = [self.warp_time_target[j] for index, (i,j) in enumerate(self.dtw_pairings[::-1]) if index in self.kept_indexes]
        warp = Warp.LinearWarp1D(self.x_in, self.x_out)
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
        time_ref, time_target = np.copy(np.array(self.x_in)).astype(int), np.copy(np.array(self.x_out)).astype(int) # DTW pairs ; two arrays of same size ; dimension: frame number
        timerange_ref, timerange_target = time_ref[-1]-time_ref[0], time_target[-1]-time_target[0]
        N = time_ref.size
        local_constraints = np.zeros((N))
        for i in range(1, N-1):
            index_x, index_y = time_ref[i]-time_ref[0], time_target[i]-time_target[0] # DTW pair number i back in index space, for cost_matrix
            center_cost = self.dtw_costmatrix[index_x, index_y] # best cost (globally)
            w_size = min(window_size, index_x, index_y, timerange_ref-index_x, timerange_target-index_y)
            upper_costs = self.dtw_costmatrix[index_x+1:index_x+w_size, index_y] + self.dtw_costmatrix[index_x, index_y+1:index_y+w_size]
            lower_costs = self.dtw_costmatrix[index_x-w_size:index_x, index_y] + self.dtw_costmatrix[index_x, index_y-w_size:index_y]
            alternative_costs = np.concatenate((upper_costs, lower_costs))
            minimal_additionnal_cost = np.min(alternative_costs) - center_cost if alternative_costs.size>0 else 0
            local_constraints[i] = max(0,minimal_additionnal_cost)
        return local_constraints


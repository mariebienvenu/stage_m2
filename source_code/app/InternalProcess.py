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

    
    def make_warp(self, feature:str=None, only_temporal=True, interpolation="linear", verbose=0):
        self.feature = feature
        curve1 = self.vanim1.find(feature)
        curve2 = self.vanim2.find(feature)
        if not only_temporal: raise NotImplementedError
        curve1.normalize()
        curve2.normalize()
        self.dtw = DynamicTimeWarping.DynamicTimeWarping(curve1, curve2) # performs the DTW algo at init time
        time_in, time_out = self.dtw.bijection
        warp = Warp.make_warp(dimension=1, interpolation=interpolation, X_in=time_in, X_out=time_out)
        return warp
    

    def make_simplified_warp(self, feature:str=None, only_temporal=True, constraint_threshold=1.5, local_scale=10, interpolation="linear", verbose=0):
        self.feature = feature
        curve1 = self.vanim1.find(feature)
        curve2 = self.vanim2.find(feature)
        if not only_temporal: raise NotImplementedError
        curve1.normalize()
        curve2.normalize()
        self.dtw = DynamicTimeWarping.DynamicTimeWarping(curve1, curve2) # performs the DTW algo at init time
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
        warp = Warp.make_warp(dimension=1, interpolation=interpolation, X_in=time_in[self.kept_indexes], X_out=time_out[self.kept_indexes])
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


    def measure_dtw_constraint(self, local=True, window_size=10): # TODO - remove, useless
        if local: return self.dtw.local_constraints(window_size)
        else : return self.dtw.global_constraints()
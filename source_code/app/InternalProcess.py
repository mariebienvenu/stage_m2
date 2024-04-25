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
        ## TODO : allow to rescale curve1 and curve2 to have same mean and std -> only smart if we want a temporal warp
        if only_temporal:
            mean1, std1 = np.mean(curve1.get_values()), np.std(curve1.get_values())
            mean2, std2 = np.mean(curve2.get_values()), np.std(curve2.get_values())
            curve2.value_transl(mean1-mean2)
            curve2.value_scale(mean1, std1/std2)
            score, pairings = m_utils.dynamic_time_warping(curve1.get_values(), curve2.get_values())
            if verbose>0: print(f"DTW achieved a score of {score}.")
            x_in = [curve1.get_times()[i] for i,j in pairings[::-1]] 
            x_out = [curve2.get_times()[j] for i,j in pairings[::-1]]
            warp = Warp.LinearWarp1D(x_in,x_out)
            return warp
        raise NotImplementedError
    
    
    def make_new_anim(self, channels:List[str]=None, warps:List[Warp.AbstractWarp]=None):
        banim2 = Animation.Animation()
        for curve in self.banim1:
            if curve.fullname in channels:
                warp = warps[channels.index(curve.fullname)]
                banim2.append(curve.apply_spatio_temporal_warp(warp, in_place=False))
            else:
                banim2.append(curve)
        return banim2



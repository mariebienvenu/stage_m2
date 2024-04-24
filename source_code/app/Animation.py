from __future__ import annotations # otherwise using Animation type hints inside Animation does not work

import os
from typing import List

import numpy as np
import pandas as pd

import app.Curve as Curve
import app.maths_utils as m_utils

class Animation(List[Curve.Curve]):

    def __init__(self, curves=[]):
        super().__init__(curves)

    def display(self, handles=True, style="markers", fig=None, row=None, col=None, doShow=False):
        for curve in self:
            fig = curve.display(handles=handles, style=style, fig=fig, row=row, col=col)
        if doShow: fig.show()
        return fig
    
    def sample(self, n_samples, start="all", stop="all"): # including stop
        '''pas "en place"'''

        if type(start) in [int, float]:
            start = [start]*len(self)
        if start in ["all", "same"]:
            start = [min([curve.time_range[0] for curve in self])]*len(self)
        if start in ["each"]:
            start = [curve.time_range[0] for curve in self]
        
        assert type(start) is list, f"Wrong type for 'start' parameter. Expected a list, got {type(start)}"
        assert len(start)==len(self), f"Wrong size for 'start' parameter. Expected {len(self)}, got {len(start)}"

        if type(stop) in [int, float]:
            stop = [stop]*len(self)
        if stop in ["all", "same"]:
            stop = [max([curve.time_range[1] for curve in self])]*len(self)
        if stop in ["each"]:
            stop = [curve.time_range[1] for curve in self]
        assert type(stop) is list, f"Wrong type for 'stop' parameter. Expected a list, got {type(stop)}"
        assert len(stop)==len(self), f"Wrong size for 'stop' parameter. Expected {len(self)}, got {len(stop)}"

        resampled_anim = Animation()
        for curve, strt, stp in zip(self, start, stop):
            times = np.concatenate((np.arange(strt, stp, step=(stp-strt)/(n_samples-1)), [stp])) if strt!=stp else np.array([strt])
            new_curve = curve.sample(times)
            resampled_anim.append(new_curve)
                
        return resampled_anim
    
    def crop(self, start=None, stop=None):
        for curve in self:
            curve.crop(start, stop)

    def __add__(self, other):
        return Animation(super().__add__(other)) #rely on List
    
    def save(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        names = [curve.fullname for curve in self]
        for i, curve in enumerate(self):
            curve.save(directory+f'/{i}.txt')

    
    def find(self, name):
        fullnames = [curve.fullname for curve in self]
        return self[fullnames.index(name)]


    @staticmethod
    def load(directory):
        res, i = Animation(), 0
        while os.path.exists(directory+f'/{i}.txt'):
            curve = Curve.Curve.load(directory+f'/{i}.txt')
            res.append(curve)
            i += 1
        return res
    

    @staticmethod
    def correlate(anim1:Animation, anim2:Animation, verbose=0):
        correlation_matrix = np.ones((len(anim1), len(anim2)), dtype=np.float64)*np.nan
        for i, curve1 in enumerate(anim1):
            for j, curve2 in enumerate(anim2):
                #assert len(curve1) == len(curve2), f"Cannot compare animation curves of different length: {len(curve1)} != {len(curve2)}"
                values1 = curve1.get_values()
                values2 = curve2.get_values()
                try:
                    correlation_matrix[i,j] = m_utils.correlation(values1, values2)
                except AssertionError as msg:
                    if verbose>0: print(f"Impossible to compute correlation between {curve1} and {curve2}. \n\t{msg}")
        rows = [curve.fullname for curve in anim1]
        columns = [curve.fullname for curve in anim2]
        dataframe = pd.DataFrame(correlation_matrix, columns=columns, index=rows)
        return dataframe


    def __mod__(self, other:Animation):
        return Animation.correlate(self, other)
    
    def check(self):
        for curve in self:
            curve.check()


    def enrich(self):
        additionnal_curves = Animation()
        for curve in self:
            features = curve.compute_features()
            additionnal_curves += Animation(features)
        self += additionnal_curves
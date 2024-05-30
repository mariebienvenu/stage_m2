from __future__ import annotations # otherwise using Animation type hints inside Animation does not work

import os
from typing import List
from collections.abc import Iterable
from enum import Enum

import numpy as np
import pandas as pd

import app.Curve as Curve
import app.maths_utils as m_utils

class TimeIndication(Enum):
    ALL = 0
    EACH = 1
    SAME = 2


class Animation(List[Curve.Curve]):

    def __init__(self, curves=[]):
        super().__init__(curves)


    def display(self, handles=True, style="markers", fig=None, row=None, col=None, doShow=False):
        for curve in self:
            fig = curve.display(handles=handles, style=style, fig=fig, row=row, col=col)
        if doShow: fig.show()
        return fig
    

    @property
    def time_range(self):
        return (min([curve.time_range[0] for curve in self]), max([curve.time_range[1] for curve in self]))
    
    
    def sample(
            self,
            n_samples:int|Iterable[int]|None=None,
            start:float|Iterable[float]|TimeIndication=TimeIndication.ALL,
            stop:float|Iterable[float]|TimeIndication=TimeIndication.ALL
        ):
        """Does not sample "in place". Stops are included as sampling times."""
        if n_samples is None: return Animation([curve.sample() for curve in self])
        try:
            enumerate(n_samples)
        except AttributeError: # "int is not iterable"
            n_samples = [n_samples]*len(self)

        for param, operator, index in zip([start, stop], [min, max], [0,1]):
            try:
                param.value
            except AttributeError: # "list does not have value", or "float does not have value"
                try:
                    enumerate(param)
                except AttributeError: # "float is not iterable"
                    param = [param]*len(self)
                finally:
                    assert len(param)==len(self), f"Wrong size for '{param._name_}' parameter. Expected {len(self)}, got {len(param)}"
            finally:
                if param in [TimeIndication.ALL, TimeIndication.SAME]:
                    param = [operator([curve.time_range[index] for curve in self])]*len(self)
                elif param is TimeIndication.EACH:
                    param = [curve.time_range[index] for curve in self]
                else:
                    raise TypeError(f"Did not provide with correct enum. Expected TimeIndication, got {type(param)}.")

        resampled_anim = Animation()
        start : Iterable = start #          -
        stop : Iterable = stop #            - > Necessary to have autocomplete on curve three lines later
        n_samples : Iterable = n_samples #  -
        for curve, strt, stp, n in zip(self, start, stop, n_samples, strict=True):
            times = np.concatenate((np.arange(strt, stp, step=(stp-strt)/(n-1)), [stp])) if strt!=stp else np.array([strt])
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


    def time_scale(self, center=0, scale=1):
        for curve in self:
            curve.time_scale(center, scale)
    
    def value_scale(self, center=0, scale=1):
        for curve in self:
            curve.value_scale(center, scale)

    def time_transl(self, translation):
        for curve in self:
            curve.time_transl(translation)

    def value_transl(self, translation):
        for curve in self:
            curve.value_transl(translation)

    def update_time_range(self):
        for curve in self:
            curve.update_time_range()
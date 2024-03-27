import numpy as np
from typing import List

from app.Curve import Curve

class Animation(List[Curve]):

    def __init__(self, curves=[]):
        super().__init__(curves)

    def display(self, handles=True, style="markers", fig=None, row=None, col=None, doShow=False):
        for curve in self:
            fig = curve.display(handles=handles, style=style, fig=fig, row=row, col=col)
        if doShow: fig.show()
        return fig
    
    def sample(self, n_samples, start="all", stop="all"):
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

        ## TODO: -- un peu ugly d'utiliser la fonction evaluate()... mais bon c'est validé par Damien
        resampled_anim = Animation()
        for curve, strt, stp in zip(self, start, stop):
            times = np.arange(strt, stp, step=(stp-strt)/n_samples) if strt!=stp else np.array([strt])
            new_curve = curve.sample(times)
            resampled_anim.append(new_curve)
            ## TODO: -- recover other informations ? if possible & useful... (probably not useful since they will be incomplete)
        
        return resampled_anim
    
    def crop(self, start=None, stop=None):
        for curve in self:
            curve.crop(start, stop)

    def __add__(self, other):
        return Animation(super().__add__(other)) #rely on List
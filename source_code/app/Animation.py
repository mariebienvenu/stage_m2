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
    
    def resample(self, n_samples, start=None, stop=None):
        # TODO check if we want a resampling for the entire anim, or if we treat curves independently (ex. diff time ranges)
        start = start if start is not None else min([curve.time_range[0] for curve in self])
        stop = stop if stop is not None else max([curve.time_range[1] for curve in self])
        times = np.expand_dims(np.arange(start, stop, step=(stop-start)/n_samples),1)

        ## TODO: -- un peu ugly d'utiliser la fonction evaluate()... mais bon c'est validé par Damien
        resampled_anim = Animation()
        for curve in self:
            values = np.expand_dims(np.array([curve.pointer.evaluate(time) for time in times]),1)
            resampled_anim.append(Curve(np.hstack((times, values)), fullname=curve.fullname))
            ## TODO: -- recover other informations ? if possible & useful... (probably not useful since they will be incomplete)
        
        return resampled_anim
    
    def crop(self, start=None, stop=None):
        for curve in self:
            curve.crop(start, stop)

    def __add__(self, other):
        return Animation(super().__add__(other)) #rely on List
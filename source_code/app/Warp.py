
import numpy as np
import scipy.interpolate as interpolate

class WarpInterpolation:
    
    def STEP(x, **kwargs):
        return kwargs.get("y0")
    
    def LINEAR(x, **kwargs):
        x_before, x_after = kwargs.get("x0"), kwargs.get("x1")
        y_before, y_after = kwargs.get("y0"), kwargs.get("y1")
        return y_before + (x_after-x)*(y_after-y_before)/(x_after-x_before)
    

class AbstractWarp:
    def __init__(self):
        raise NotImplementedError
    def __call__(self, x):
        raise NotImplementedError



class LinearWarp1D(AbstractWarp):

    def __init__(self, X_in, X_out):
        #order = np.array(X_in).argsort()
        self.X = np.array(X_in)#[order]
        self.Y = np.array(X_out)#[order]

    def __call__(self, t, x):
        return np.interp(t, self.X, self.Y),x
    

class LinearWarp2D(AbstractWarp):

    def __init__(self, X_in, Y_in, X_out, Y_out):
        self.points = np.vstack((np.array(X_in), np.array(Y_in))).T
        self.values = np.vstack((np.array(X_out), np.array(Y_out))).T

        self.interpolator = interpolate.LinearNDInterpolator(self.points, self.values)

    def __call__(self, x, y):
        results = self.interpolator(np.array(x), np.array(y))
        return results[:,0], results[:,1]


class Warp:

    def __init__(self, X, Y, interpolation=WarpInterpolation.LINEAR):
        order = np.array(X).argsort()
        self.X = np.array(X)[order]
        self.Y = np.array(Y)[order]
        self.interpolation = interpolation
        try:
            interpolation(0)
        except AttributeError: #typically, string has no call method
            self.interpolation = getattr(WarpInterpolation,interpolation)

    
    def find_interval(self,x):
        if x < self.X[0]:
            return -1
        if x > self.X[-1]:
            return self.X.size
        index = 1
        while x > self.X[index]:
            index +=1
        return index-1
    

    def __call__(self, x):
        index = self.find_interval(x)
        if x<0:
            return self.Y[0]
        elif x>self.X.size-1:
            return self.Y[-1]
        
        x0, y0, x1, y1 = self.X[index], self.Y[index], self.X[index+1], self.Y[index+1]
        return self.interpolation(x, x0=x0, y0=y0, x1=x1, y1=y1)

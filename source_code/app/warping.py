
import numpy as np
import scipy.interpolate as interpolate

class AbstractWarp:
    def __init__(self, input_data, output_data):
        self.input_data, self.output_data = input_data, output_data
    
    def __call__(self, x):
        raise NotImplementedError
    
    def __repr__(self):
        return f"{type(self).__name__}({self.input_data.shape},{self.output_data.shape})"


class LinearWarp1D(AbstractWarp):

    def __init__(self, X_in, X_out):
        super(LinearWarp1D, self).__init__(np.array(X_in), np.array(X_out))

    def __call__(self, t, x):
        return np.interp(t, self.input_data, self.output_data),x
    

class CubicWarp1D(AbstractWarp):

    def __init__(self, X_in, X_out):
        super(CubicWarp1D, self).__init__(np.array(X_in), np.array(X_out))
        self.interpolator = interpolate.CubicSpline(self.input_data, self.output_data, extrapolate=False, bc_type="natural") #safer

    def __call__(self, t, x):
        return self.interpolator(t),x


class LinearWarp2D(AbstractWarp):

    def __init__(self, X_in, Y_in, X_out, Y_out):
        input_data = np.vstack((np.array(X_in), np.array(Y_in))).T
        output_data = np.vstack((np.array(X_out), np.array(Y_out))).T
        super(LinearWarp2D, self).__init__(input_data, output_data)
        self.interpolator = interpolate.LinearNDInterpolator(self.input_data, self.output_data)

    def __call__(self, x, y):
        results = self.interpolator(np.array(x), np.array(y))
        return results[:,0], results[:,1]
    


def make_warp(dimension=1, interpolation="linear", **kwargs): ## this is the function that should be called from outside
    if dimension==1 and interpolation=="linear":
        return LinearWarp1D(**kwargs)
    elif dimension==1 and interpolation=="cubic":
        return CubicWarp1D(**kwargs)
    elif dimension==2 and interpolation=="linear":
        return LinearWarp2D(**kwargs)
    raise NotImplementedError(f"The requested warp is not implemented yet. Asked: dimension={dimension}, interpolation={interpolation}")
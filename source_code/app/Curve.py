
import os
from enum import Enum, unique
from copy import deepcopy

import numpy as np
import plotly.graph_objects as go
import bpy

from app.color import Color
import app.maths_utils as m_utils
import app.visualisation as vis


@unique
class Easing_Mode(Enum):
    AUTO = 0 # most used
    EASE_IN = 1
    EASE_OUT = 2
    EASE_IN_OUT = 3

@unique
class Handle_Type(Enum):
    FREE = 0
    VECTOR = 1
    ALIGNED = 2
    AUTO = 3
    AUTO_CLAMPED = 4 # most used

@unique
class Interpolation_Type(Enum):
    CONSTANT = 0
    LINEAR = 1
    BEZIER = 2 # most used
    SINE = 3
    QUAD = 4
    CUBIC = 5
    QUART = 6
    QUINT = 7
    EXPO = 8
    CIRC = 9
    BACK = 10
    BOUNCE = 11
    ELASTIC = 12

@unique
class Key_Type(Enum):
    KEYFRAME = 0 # most used
    BREAKDOWN = 1
    MOVING_HOLD = 2
    EXTREME = 3
    JITTER = 4

@unique
class Attributes_Name(Enum): # ordered !
    id = 0
    time = 1
    value = 2
    easing_mode = 3 # TODO Curve.Attributes_Name has ugly formatting...
    handle_left_x = 4
    handle_left_y = 5
    handle_left_type = 6
    handle_right_x = 7
    handle_right_y = 8
    handle_right_type = 9
    interpolation = 10
    key_type = 11
    amplitude = 12
    back = 13
    period = 14


class Curve:

    N_ATTRIBUTES = len(list(Attributes_Name))
    ATTRIBUTE_TYPES = [int, np.float64, np.float64, Easing_Mode, np.float64, np.float64, Handle_Type, np.float64, np.float64, Handle_Type, Interpolation_Type, Key_Type, np.float64, np.float64,  np.float64] # size 15
    ARE_ENUMS = [False, False, False, True, False, False, True, False, False, True, True, True, False, False, False]
    DEFAULT_ATTRIBUTES = np.array([Easing_Mode.AUTO.value, 0, 0, Handle_Type.AUTO.value, 0, 0, Handle_Type.AUTO.value, Interpolation_Type.BEZIER.value, Key_Type.KEYFRAME.value, 0, 0, 0]) # no default for id, key, value -> size 12
    
    def __init__(
            self,
            coordinates=np.zeros((1,2), dtype=np.float64), # shape (N,2)
            easing_mode = Easing_Mode.AUTO,
            tangent_left_handle_x = np.float64(0),
            tangent_left_handle_y = np.float64(0),
            tangent_left_type=Handle_Type.AUTO,
            tangent_right_handle_x = np.float64(0),
            tangent_right_handle_y = np.float64(0),
            tangent_right_type=Handle_Type.AUTO,
            interpolation=Interpolation_Type.BEZIER,
            key_type=Key_Type.KEYFRAME,
            amplitude=np.float64(0),
            back=np.float64(0),
            period=np.float64(0),
            fullname='',
            time_range=None,
            **kwargs
        ):

        coordinates = np.array(coordinates)
        assert len(coordinates.shape)==2, f"Wong shape. Coordinates array is {len(coordinates.shape)} dimensional instead of 2."
        assert coordinates.shape[1]==2, f"Wrong shape. Coordinates array should have 2 columns (time, value), got {coordinates.shape[1]} columns instead."

        ids = np.expand_dims(np.array(list(range(coordinates.shape[0]))),1)

        to_stack = [ids, coordinates]

        additionnal_attributes = [easing_mode, tangent_left_handle_x, tangent_left_handle_y, tangent_left_type, tangent_right_handle_x, tangent_right_handle_y, tangent_right_type, interpolation, key_type, amplitude, back, period]
        for i, (input_arg, is_enum) in enumerate(zip(additionnal_attributes, Curve.ARE_ENUMS[3:])): # try a try enumerate(input_arg) to test ?
            try:
                enumerate(input_arg)
                input_arg = np.array(input_arg)
                if len(input_arg.shape)==1:
                    input_arg = np.expand_dims(input_arg, axis=1)
            except TypeError: # object not iterable
                input_arg = np.ones_like(ids)*(input_arg.value if is_enum else input_arg)
            to_stack.append(np.copy(input_arg))
        
        self.array = np.hstack(to_stack) # shape (n, N_ATTRIBUTES)
        self.check()

        self.fullname, self.time_range = fullname, time_range
        if time_range is None:
            self.time_range = (np.min(coordinates[:, 0]), np.max(coordinates[:, 0]))
        
        for (key, value) in kwargs.items():
            assert key not in dir(self), f"Invalid name: {key} is already taken."
            if value is not None :
                self.__setattr__(key, value) # no autocomplete for these

    def check(self):
        assert len(self.array.shape)==2 and self.array.shape[1] == Curve.N_ATTRIBUTES, f"Curve self checking failed: array shape is {self.array.shape} instead of (N, {Curve.N_ATTRIBUTES})"
        assert self.array.dtype == np.float64, f"Curve self checking failed: array type is {self.array.dtype} instead of np.float64."
        for column, (expected_type, is_enum) in enumerate(zip(Curve.ATTRIBUTE_TYPES, Curve.ARE_ENUMS)):
            if is_enum:
                column_content = self._get_column(column)
                assert np.sum(np.abs(column_content-column_content.astype(np.int32))) < 1e-3, f"Curve self checking failed: found non-integers in column {column} ('{expected_type.__name__}')."
                for element in column_content:
                    try:
                        expected_type(int(element))
                    except ValueError as msg:
                        raise AssertionError(f"Curve self checking failed: found wrong element in column {column}, {msg}")

    def __len__(self):
        return self.array.shape[0]

    def get_keyframe(self, id):
        kf = self._get_row(id)
        assert kf[0] == id, "Error in getting keyframe."
        return kf[1,:]
    
    def set_keyframe(self, id, array): # array SHOULD start with id
        assert id==array[0], "Wrong ID !"
        kf = self._get_row(id)
        assert kf[0] == id, "Error in finding keyframe."
        self._set_row(id, array)
    
    def get_attribute(self, attribute_name:Attributes_Name|str):
        try :
            column = attribute_name.value
        except AttributeError:
            column = Attributes_Name[attribute_name].value
        return self._get_column(column)
    
    def set_attribute(self, attribute_name:Attributes_Name|str, array):
        try :
            column = attribute_name.value
        except AttributeError:
            column = Attributes_Name[attribute_name].value
        self._set_column(column, array)
    
    def time_scale(self, center=0, scale=1):
        for attr in [Attributes_Name.time, Attributes_Name.handle_left_x, Attributes_Name.handle_right_x]:
            self.set_attribute(attr, (self.get_attribute(attr)-center)*scale + center)
        self.update_time_range()

    def value_scale(self, center=0, scale=1):
        for attr in [Attributes_Name.value, Attributes_Name.handle_left_y, Attributes_Name.handle_right_y]:
            self.set_attribute(attr, (self.get_attribute(attr)-center)*scale + center)

    def time_transl(self, translation):
        for attr in [Attributes_Name.time, Attributes_Name.handle_left_x, Attributes_Name.handle_right_x]:
            self.set_attribute(attr, self.get_attribute(attr)+translation)
        self.update_time_range()

    def value_transl(self, translation):
        for attr in [Attributes_Name.value, Attributes_Name.handle_left_y, Attributes_Name.handle_right_y]:
            self.set_attribute(attr, self.get_attribute(attr)+translation)

    def normalize(self, mean=0, std=1):
        current_mean, current_std = np.mean(self.get_values()), np.std(self.get_values())
        self.value_transl(mean-current_mean)
        self.value_scale(mean, std/current_std)

    
    def apply_spatio_temporal_warp(self, warp, in_place=True):
        # warp is a function that takes 2 arguments (t,value) and returns two values (t', value'). Handles arrays of same size.
        obj = self if in_place else deepcopy(self)
        firsts = [Attributes_Name.time, Attributes_Name.handle_left_x, Attributes_Name.handle_right_x]
        seconds = [Attributes_Name.value, Attributes_Name.handle_left_y, Attributes_Name.handle_right_y]
        for first, second in zip(firsts, seconds):
            t, x = obj.get_attribute(first), obj.get_attribute(second)
            t_prime, x_prime = warp(t,x)
            obj.set_attribute(first, t_prime)
            obj.set_attribute(second, x_prime)
        return obj


    def add_keyframe(self, time, value):
        new_id = self.array.shape[0]
        new_row = np.expand_dims(np.concatenate((np.array([new_id, time, value]), np.copy(Curve.DEFAULT_ATTRIBUTES))), axis=0)
        array = np.vstack((self.array, new_row))
        self.array = np.copy(array)

    def move_keyframe(self, id, new_time, new_value):
        self.array[id, 1:3] = new_time, new_value


    def display(self, handles=True, style="markers", color=None, opacity=1., name=None, fig=None, row=None, col=None, doShow=False):
        if color is None: color = getattr(self, 'color', Color.next())
        if name is None: name = self.fullname
        times = self.get_times()
        values = self.get_values()
        if not handles:
            fig = vis.add_curve(x=times, y=values, name=name, style=style, color=color, opacity=opacity, fig=fig, row=row, col=col)
        if handles:
            handle_times = np.concatenate((self.get_attribute('handle_left_x'), self.get_attribute('handle_right_x')))
            handle_values = np.concatenate((self.get_attribute('handle_left_y'), self.get_attribute('handle_right_y')))
            for id in range(len(self)): # drawing the keyframes with tangent handles
                x = [handle_times[id], times[id], handle_times[id+len(self)]]
                y = [handle_values[id], values[id], handle_values[id+len(self)]]
                fig = vis.add_curve(x=x, y=y, style="markers+lines", color=color, opacity=0.4*opacity, fig=fig, row=row, col=col, legend=False)
            fig = self.sample().display(handles=False, style='lines', color=color, opacity=opacity, name=name, fig=fig, row=row, col=col) # drawing the interpolated curve
        if doShow: fig.show()
        return fig
    
    def get_keyframe_attribute(self, id, attribute_name:Attributes_Name|str):
        column = attribute_name.value if type(attribute_name) is Attributes_Name else Attributes_Name[attribute_name].value
        expected_type = Curve.ATTRIBUTE_TYPES[column]
        return expected_type(self.array[id, column])
    
    def set_keyframe_attribute(self, id, attribute_name:Attributes_Name|str, value):
        column = attribute_name.value if type(attribute_name) is Attributes_Name else Attributes_Name[attribute_name].value
        expected_type = Curve.ATTRIBUTE_TYPES[column]
        if not Curve.ARE_ENUMS[column]:
            assert type(value) == expected_type, f"Wrong Type. Expected {expected_type}, got {type(value)}."
            self.array[id, column] = value
            return
        try:
            self.array[id, column] = value.value
        except AttributeError: #typically, "string has no value attribute"
            self.array[id, column] = expected_type[value].value

    def rename(self, new_name=""):
        self.fullname = new_name
        
    def _get_row(self, row):
        return np.copy(self.array[row,:])
    
    def _get_column(self, column):
        return np.copy(self.array[:,column])
    
    def _set_row(self, row, array):
        self.array[row,:] = np.copy(array)

    def _set_column(self, column, array):
        self.array[:,column] = np.copy(array)

    @staticmethod
    def from_array(array, **kwargs):
        curve = Curve(**kwargs)
        curve.array = np.copy(array)
        curve.check()
        curve.update_time_range()
        return curve
    
    def __repr__(self):
        n = f', "{self.fullname}"' if len(self.fullname)>0 else ''
        r = f', ({self.time_range[0]:.1f},{self.time_range[1]:.1f})'
        return f"Curve({len(self)}{n}{r})"

    def get_values(self):
        return self.get_attribute(Attributes_Name.value)
    
    def get_times(self):
        return self.get_attribute(Attributes_Name.time)
    
    def update_time_range(self):
        times = self.get_times()
        self.time_range = (np.min(times), np.max(times))

    def get_value_range(self):
        values = self.get_values()
        return (np.min(values), np.max(values))

    def crop(self, start=None, stop=None):
        start = start if start is not None else self.time_range[0]
        stop = stop if stop is not None else self.time_range[1]
        indexes = []
        for i, time in enumerate(self.get_times()):
            if time >= start and time <= stop:
                indexes.append(i)
        self.array = np.copy(self.array[indexes,:])
        self.update_time_range()

    def get_keyframe_derivatives(self):
        ## computes derivatives at keyframes. Broken handles are automatically assigned a derivative of 0.
        dx = self.get_attribute(Attributes_Name.handle_right_x) - self.get_attribute(Attributes_Name.handle_left_x)
        dy = self.get_attribute(Attributes_Name.handle_right_y) - self.get_attribute(Attributes_Name.handle_left_y)
        left_unbroken_handles = np.array([ht not in [Handle_Type.FREE.value, Handle_Type.VECTOR.value] for ht in self.get_attribute(Attributes_Name.handle_left_type)])
        right_unbroken_handles = np.array([ht not in [Handle_Type.FREE.value, Handle_Type.VECTOR.value] for ht in self.get_attribute(Attributes_Name.handle_right_type)])
        return dy/dx*left_unbroken_handles*right_unbroken_handles 
    
    def sample(self, times:np.ndarray|int=None):
        if len(self) <= 1 : return self ## TODO maybe test if this is a good idea ? Added in a debug session with Main having no blender
        assert "pointer" in dir(self), "Cannot sample a curve with no associated fcurve."
        if times is None: times = int(self.time_range[1]-self.time_range[0]+1)
        try:
            enumerate(times)
        except TypeError: # typically, 'int' object is not iterable
            times = np.linspace(self.time_range[0], self.time_range[1], times)
        values = np.array([self.pointer.evaluate(time) for time in times]) # un peu ugly d'utiliser la fonction evaluate() mais c'est validé par Damien
        coordinates = np.vstack((times, values)).T
        color = getattr(self, "color", None) # here we recover additionnal info on the curve that we would like to forward to the new, sampled curve
        new_curve = Curve(coordinates, fullname=self.fullname+" sampled" if len(self.fullname)>0 else "sampled", color=color)
        new_curve.update_time_range()
        return new_curve
    
    def save(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
        np.savetxt(filename, X=self.array, header=self.fullname+'\n')

    @staticmethod
    def load(filename):
        assert os.path.exists(filename), f"Unable to load content of '{filename}' as it does not exist."
        file = open(filename)
        content = np.loadtxt(filename)
        if len(content.shape)==1:
            content = np.expand_dims(content, axis=0)
        curve = Curve.from_array(content)
        curve.rename(file.readline()[2:-1]) #remove "# ", and "\n" at the end
        curve.update_time_range()
        file.close()
        return curve
    

    def _derivative(self, finite_scheme=None, name='', n_samples=None):
        # Assumes that either self has an associated blender pointer, or is uniformally sampled
        
        try:
            fcurve : bpy.types.FCurve = self.pointer
        except AttributeError:
            step = (self.time_range[1]-self.time_range[0])/(len(self))
            derivative = finite_scheme(self.get_values(), step)
            co = np.vstack((self.get_times()[1:-1], derivative)).T
            return Curve(coordinates=co, fullname=f'{name} of {self.fullname}')
        
        n_samples = n_samples if n_samples is not None else len(self)
        step = (self.time_range[1]-self.time_range[0])/(n_samples-1)
        delta_t = step/20
        
        sampling_t = [self.time_range[0] + i*step for i in range(n_samples)]
        sampling_v = [fcurve.evaluate(t-delta_t) if i%3==0 else (fcurve.evaluate(t) if i%3==1 else fcurve.evaluate(t+delta_t))  for (i,t) in enumerate([sampling_t[j//3] for j in range(3*n_samples)])]
        derivative = finite_scheme(sampling_v, delta_t)
        derivative = derivative[::3]
        co = np.vstack((sampling_t, derivative)).T
        return Curve(coordinates=co, fullname=f'{name} of {self.fullname}')
    

    def first_derivative(self, n_samples=None):
        # n_samples is only used if the curve can be sampled i.e. if we have some FCurve pointer
        return self._derivative(m_utils.derivee, name='First derivative', n_samples=n_samples)
    
    def second_derivative(self, n_samples=None):
        return self._derivative(m_utils.derivee_seconde, name='Second derivative', n_samples=n_samples)
        
        
    def get_auto_crop(self, use_handles=True, default_value=0, threshold=0.1, padding_out=10, padding_in=3, patience=0):
        # arguments after "use_handles" only need to be specified when "use_handles" is set to False
        # argument "threshold" is a percentage of the value range, between 0 and 1
        times, values = self.get_times(), self.get_values()
        if use_handles:
            order = np.argsort(self.get_times())
            times = times[order]
            values = values[order]
            derivatives = self.get_keyframe_derivatives()[order]
            zipped = zip(values, values[::-1], derivatives, derivatives[::-1])
            start = 1
            stop = len(self)-2
            for i, (value, opp_value, derivative, opp_derivative) in enumerate(zipped):
                if start == i and derivative==0 and value==values[i-1]:
                    start += 1
                if stop == len(self)-1-i and opp_derivative==0 and opp_value==values[-i]:
                    stop -= 1
            if start==len(self) or stop==0:
                return (times[0], times[0]) # all keyframes have same value and tangents -> only one keyframe is enough
            start -= 1
            stop += 1
        else:
            start = padding_out
            stop = len(self)-padding_out
            used_patience_left = 0
            used_patience_right = 0
            value_range = self.get_value_range()[1]-self.get_value_range()[0]
            for i, (value, opp_value) in enumerate(zip(values, values[::-1])):
                if start == i:
                    if not abs(value-default_value)<threshold*value_range and used_patience_left<patience:
                        start += 1
                        used_patience_left += 1
                    elif abs(value-default_value)<threshold*value_range:
                        start += 1
                        used_patience_left = 0
                if stop == len(self)-i:
                    if not abs(opp_value-default_value)<threshold*value_range and used_patience_right<patience:
                        stop -= 1
                        used_patience_right += 1
                    elif abs(opp_value-default_value)<threshold*value_range:
                        stop -= 1
                        used_patience_right = 0
            if start==len(self) or stop==0:
                return (times[0], times[0]) # found nothing good to keep...
            start -= padding_in + used_patience_left
            stop += padding_in + used_patience_right
            start = max(0, start)
            stop = min(stop, len(self))
        assert start <= stop, "Problem encountered when autocropping."
        return (times[start], times[stop])
        

    def compute_features(self):
        return [self.first_derivative(), self.second_derivative()]
    

    def __deepcopy__(self, memo):
        new = Curve.from_array(self.array.__deepcopy__(memo), fullname=self.fullname, color=getattr(self, "color", None))
        return new
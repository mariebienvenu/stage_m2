
import os
from enum import Enum, unique

import numpy as np
import plotly.graph_objects as go

from app.Color import Color

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
    easing_mode = 3
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
            except TypeError: # object not iterable
                input_arg = np.ones_like(ids)*(input_arg.value if is_enum else input_arg)
            #input_arg = np.array(input_arg) if type(input_arg)!=Curve.ATTRIBUTE_TYPES[i+3] else np.ones_like(ids)*(input_arg.value if is_enum else input_arg)
            to_stack.append(np.copy(input_arg))
        
        self.array = np.hstack(to_stack) # shape (n, N_ATTRIBUTES)
        assert len(self.array.shape)==2, f"Wrong shape. Array is {len(self.array.shape)} dimensional instead of 2."
        assert self.array.shape[1] == Curve.N_ATTRIBUTES, f"Wrong size in initialization: {self.array.shape[1]} instead of {Curve.N_ATTRIBUTES}"

        self.fullname, self.time_range = fullname, time_range
        if time_range is None:
            self.time_range = (np.min(coordinates[:, 0]), np.max(coordinates[:, 0]))
        
        for (key, value) in kwargs.items():
            assert key not in dir(self), f"Invalid name: {key} is already taken."
            self.__setattr__(key, value) # no autocomplete for these
        

    def get_keyframe(self, id):
        kf = self._get_row(id)
        assert kf[0] == id, "Error in getting keyframe."
        return kf[1,:]
    
    def get_attribute(self, attribute_name:Attributes_Name|str):
        column = attribute_name.value if type(attribute_name) is Attributes_Name else Attributes_Name[attribute_name].value
        return self._get_column(column)
    
    
    def time_scale(self, center=0, scale=1):
        self.array[:,1] = (self.array[:,1]-center)*scale + center

    def value_scale(self, center=0, scale=1):
        self.array[:,2] = (self.array[:,2]-center)*scale + center

    def time_transl(self, translation):
        self.array[:,1] += translation

    def value_transl(self, translation):
        self.array[:,2] += translation


    def add_keyframe(self, time, value):
        new_id = self.array.shape[0]
        new_row = np.expand_dims(np.concatenate((np.array([new_id, time, value]), np.copy(Curve.DEFAULT_ATTRIBUTES))), axis=0)
        array = np.vstack((self.array, new_row))
        self.array = np.copy(array)

    def move_keyframe(self, id, new_time, new_value):
        self.array[id, 1:3] = new_time, new_value


    def display(self, handles=True, style="markers", color=None, fig=None, row=None, col=None, doShow=False):
        if fig is None:
            fig = go.Figure()
        times = self.get_times()
        values = self.get_values()
        color = self.color if 'color' in dir(self) else (color if color is not None else Color.next())
        fig.add_trace(go.Scatter(x=times, y=values, name=self.fullname, mode=style, marker_color=f'rgb{color}'), row=row, col=col) # drawing the keyframes
        if handles:
            handle_times = np.concatenate((self.get_attribute('handle_left_x'), self.get_attribute('handle_right_x')))
            handle_values = np.concatenate((self.get_attribute('handle_left_y'), self.get_attribute('handle_right_y')))
            for id in range(len(self)): # drawing the handles
                x = [handle_times[id], times[id], handle_times[id+len(self)]]
                y = [handle_values[id], values[id], handle_values[id+len(self)]]
                fig.add_trace(go.Scatter(x=x, y=y, mode='markers+lines', marker_color=f'rgba{color+tuple([0.4])}', showlegend=False), row=row, col=col)

        if doShow: fig.show()
        return fig
    
    def check(self):
        assert len(self.array.shape)==2 and self.array.shape[1] == Curve.N_ATTRIBUTES, f"Self checking failed ! Current shape is {self.array.shape} instead of (N, {self.N_ATTRIBUTES+1})"
        # TODO: Curve.check() should also check types of columns

    def __len__(self):
        return self.array.shape[0]
    
    def get_keyframe_attribute(self, id, attribute_name:Attributes_Name|str):
        column = attribute_name.value if type(attribute_name) is Attributes_Name else Attributes_Name[attribute_name].value
        expected_type = Curve.ATTRIBUTE_TYPES[column]
        return expected_type(self.array[id, column])
    
    def set_keyframe_attribute(self, id, attribute_name:Attributes_Name|str, value):
        column = attribute_name.value if type(attribute_name) is Attributes_Name else Attributes_Name[attribute_name].value
        expected_type = Curve.ATTRIBUTE_TYPES[column]
        assert type(value) == expected_type, f"Wrong Type. Expected {expected_type}, got {type(value)}."
        self.array[id, column] = value.value if Curve.ARE_ENUMS[column] else value

    def rename(self, new_name=""):
        self.fullname = new_name
        
    def _get_row(self, row):
        return np.copy(self.array[row,:])
    
    def _get_column(self, column):
        return np.copy(self.array[:,column])

    @staticmethod
    def from_array(array, **kwargs):
        curve = Curve(**kwargs)
        curve.array = np.copy(array)
        curve.check()
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

    def crop(self, start=None, stop=None):
        start = start if start is not None else self.time_range[0]
        stop = stop if stop is not None else self.time_range[1]
        indexes = []
        for i, time in enumerate(self.get_times()):
            if time >= start and time <= stop:
                indexes.append(i)
        self.array = np.copy(self.array[indexes,:])
        self.update_time_range()

    def get_derivatives(self): # TODO untested in curve_and_animations.py i think
        ## TODO : c'est FAUX dans le cas d'un handle de type différent !!! -> update: c'est changé mais toujours pas top
        dx = self.get_attribute(Attributes_Name.handle_right_x) - self.get_attribute(Attributes_Name.handle_left_x)
        dy = self.get_attribute(Attributes_Name.handle_right_y) - self.get_attribute(Attributes_Name.handle_left_y)
        left_unbroken_handles = np.array([ht not in [Handle_Type.FREE.value, Handle_Type.VECTOR.value] for ht in self.get_attribute(Attributes_Name.handle_left_type)])
        right_unbroken_handles = np.array([ht not in [Handle_Type.FREE.value, Handle_Type.VECTOR.value] for ht in self.get_attribute(Attributes_Name.handle_right_type)])
        return dy/dx*left_unbroken_handles*right_unbroken_handles # broken handles are automatically assigned a derivative of 0
    
    def sample(self, times):
        assert "pointer" in dir(self), "Cannot sample a curve with no associated fcurve."
        values = np.array([self.pointer.evaluate(time) for time in times])
        coordinates = np.vstack((times, values)).T
        new_curve = Curve(coordinates, fullname=self.fullname+" sampled" if len(self.fullname)>0 else "sampled")
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
        
        
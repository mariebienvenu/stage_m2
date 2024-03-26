import bpy
import numpy as np

import app.Curve, app.Animation

import importlib
importlib.reload(app.Curve)
importlib.reload(app.Animation)

from app.Curve import Easing_Mode, Handle_Type, Interpolation_Type, Key_Type, Curve
from app.Animation import Animation


def list_objects():
    for o in bpy.data.objects:
        print(o.name, o.location)

def list_scenes():
    # print all scene names in a list
    print(bpy.data.scenes.keys())

    
def get_animation(obj_name):
    animation = Animation()

    obj = None
    for o in bpy.data.objects:
        if o.name == obj_name:
            obj = o
    if obj is None:
        return animation
    
    for curve in obj.animation_data.action.fcurves:
        
        data = np.zeros((len(curve.keyframe_points), Curve.N_ATTRIBUTES))
        for i, kf in enumerate(curve.keyframe_points):

            data[i,:] = [
                i,
                kf.co[0],
                kf.co[1],
                Easing_Mode[kf.easing].value,
                kf.handle_left[0],
                kf.handle_left[1],
                Handle_Type[kf.handle_left_type].value,
                kf.handle_right[0],
                kf.handle_right[1],
                Handle_Type[kf.handle_right_type].value,
                Interpolation_Type[kf.interpolation].value,
                Key_Type[kf.type].value,
                kf.amplitude,
                kf.back,
                kf.period
            ]

        animation.append(Curve.from_array(
            data,
            fullname=f"{curve.data_path} {['X','Y','Z'][curve.array_index]}",
            name=curve.data_path,
            channel=curve.array_index,
            time_range=tuple(curve.range()),
            pointer=curve,
            color=tuple(curve.color)
        ))

        a=1
    
    return animation

def get_crop(curve:Curve):
    order = np.argsort(curve.get_times())
    times = curve.get_times()[order]
    values = curve.get_values()[order]
    derivatives = curve.get_derivatives()[order]
    zipped = zip(values, values[::-1], derivatives, derivatives[::-1])

    start = 1
    stop = len(curve)-2
    for i, (value, opp_value, derivative, opp_derivative) in enumerate(zipped):
        if start == i and derivative==0 and value==values[i-1]:
            start += 1
        if stop == len(curve)-1-i and opp_derivative==0 and opp_value==values[-i]:
            stop -= 1
    
    start -= 1
    stop += 1
    assert start <= stop, "Problem encountered when autocropping."
    return (times[start], times[stop])
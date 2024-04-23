
import bpy
import numpy as np

import app.Animation

import importlib
importlib.reload(app.Animation)

import app.Animation as Animation
from app.Animation import Curve


def list_objects():
    for o in bpy.data.objects:
        print(o.name, o.location)

def list_scenes():
    # print all scene names in a list
    print(bpy.data.scenes.keys())

    
def get_animation(obj_name):
    animation = Animation.Animation()

    obj = None
    for o in bpy.data.objects:
        if o.name == obj_name:
            obj = o
    if obj is None:
        return animation
    
    for curve in obj.animation_data.action.fcurves:
        
        data = np.zeros((len(curve.keyframe_points), Curve.Curve.N_ATTRIBUTES))
        for i, kf in enumerate(curve.keyframe_points):

            data[i,:] = [
                i,
                kf.co[0],
                kf.co[1],
                Curve.Easing_Mode[kf.easing].value,
                kf.handle_left[0],
                kf.handle_left[1],
                Curve.Handle_Type[kf.handle_left_type].value,
                kf.handle_right[0],
                kf.handle_right[1],
                Curve.Handle_Type[kf.handle_right_type].value,
                Curve.Interpolation_Type[kf.interpolation].value,
                Curve.Key_Type[kf.type].value,
                kf.amplitude,
                kf.back,
                kf.period
            ]

        animation.append(Curve.Curve.from_array(
            data,
            fullname=f"{curve.data_path} {['X','Y','Z'][curve.array_index]}",
            name=curve.data_path,
            channel=curve.array_index,
            time_range=tuple(curve.range()),
            pointer=curve,
            color=tuple(curve.color)
        ))
    
    return animation

    
def set_animation(obj_name:str, animation:Animation.Animation):
    obj = None
    for o in bpy.data.objects:
        if o.name == obj_name:
            obj = o
    if obj is None:
        return animation
    
    are_enums = Curve.Curve.ARE_ENUMS
    
    for curve in obj.animation_data.action.fcurves:

        fullname = f"{curve.data_path} {['X','Y','Z'][curve.array_index]}"
        target_curve = animation.find(fullname)

        a = list(curve.keyframe_points)

        assert len(target_curve)==len(list(curve.keyframe_points)), "Problem ! Number of keyframes changed."

        attr_names = Curve.Attributes_Name
        
        for i, kf in enumerate(curve.keyframe_points):
            
            getter = lambda x : target_curve.get_keyframe_attribute(i, x) if not are_enums[x.value] else target_curve.get_keyframe_attribute(i, x).name

            kf.co = (getter(attr_names.time), getter(attr_names.value))
            kf.easing = getter(attr_names.easing_mode)
            kf.handle_left = (getter(attr_names.handle_left_x), getter(attr_names.handle_left_y))
            kf.handle_left_type = getter(attr_names.handle_left_type)
            kf.handle_right = (getter(attr_names.handle_right_x), getter(attr_names.handle_right_y))
            kf.handle_right_type = getter(attr_names.handle_right_type)
            kf.interpolation = getter(attr_names.interpolation)
            kf.type = getter(attr_names.key_type)
            kf.amplitude = getter(attr_names.amplitude)
            kf.back = getter(attr_names.back)
            kf.period = getter(attr_names.period)

'''
def get_crop(curve:Curve.Curve): # LEGACY, moved to Curve class
    order = np.argsort(curve.get_times())
    times = curve.get_times()[order]
    values = curve.get_values()[order]
    derivatives = curve.get_keyframe_derivatives()[order]
    zipped = zip(values, values[::-1], derivatives, derivatives[::-1])

    start = 1
    stop = len(curve)-2
    for i, (value, opp_value, derivative, opp_derivative) in enumerate(zipped):
        if start == i and derivative==0 and value==values[i-1]:
            start += 1
        if stop == len(curve)-1-i and opp_derivative==0 and opp_value==values[-i]:
            stop -= 1
    if start==times.size or stop==0:
        return (times[0], times[0]) # all keyframes have same value and tangents -> only one keyframe is enough
    start -= 1
    stop += 1
    assert start <= stop, "Problem encountered when autocropping."
    return (times[start], times[stop])
'''
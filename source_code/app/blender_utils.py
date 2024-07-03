
import bpy
import numpy as np

import app.animation

import importlib
importlib.reload(app.animation)

from app.animation import Animation
import app.curve as C
from app.curve import Curve


def list_objects():
    for o in bpy.data.objects:
        print(o.name, o.location)

def list_scenes():
    # print all scene names in a list
    print(bpy.data.scenes.keys())


def get_filename():
    return bpy.data.filepath.replace('\\', '/')


def check_filename(filename):
    assert get_filename()==filename, f"Wrong file opened. Expected {filename} and found {get_filename()}."

    
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
                C.Easing_Mode[kf.easing].value,
                kf.handle_left[0],
                kf.handle_left[1],
                C.Handle_Type[kf.handle_left_type].value,
                kf.handle_right[0],
                kf.handle_right[1],
                C.Handle_Type[kf.handle_right_type].value,
                C.Interpolation_Type[kf.interpolation].value,
                C.Key_Type[kf.type].value,
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
    
    return animation


def add_key(obj_name:str, curve_name:str, n_keys=1):
    obj = None
    for o in bpy.data.objects:
        if o.name == obj_name:
            obj = o
    if obj is None:
        return
    
    for curve in obj.animation_data.action.fcurves:
        fullname = f"{curve.data_path} {['X','Y','Z'][curve.array_index]}"
        debig=0
        if curve_name == fullname:
            frames = [kf.co[0] for kf in curve.keyframe_points]
            for i in range(n_keys):
                obj.keyframe_insert(curve.data_path, index=curve.array_index, frame=max(frames)+1+i)


def remove_key(obj_name:str, curve_name:str, n_keys=1):
    obj = None
    for o in bpy.data.objects:
        if o.name == obj_name:
            obj = o
    if obj is None:
        return
    
    for curve in obj.animation_data.action.fcurves:
        fullname = f"{curve.data_path} {['X','Y','Z'][curve.array_index]}"
        debug=0
        if curve_name == fullname:
            frames = [kf.co[0] for kf in curve.keyframe_points]
            for i in range(n_keys):
                obj.keyframe_delete(curve.data_path, index=curve.array_index, frame=frames[-1-i])

    
def set_animation(obj_name:str, animation:Animation):
    obj = None
    for o in bpy.data.objects:
        if o.name == obj_name:
            obj = o
    if obj is None:
        return animation
    
    are_enums = Curve.ARE_ENUMS
    
    for curve in obj.animation_data.action.fcurves:

        fullname = f"{curve.data_path} {['X','Y','Z'][curve.array_index]}"
        target_curve = animation.find(fullname)

        target_size = len(target_curve)
        current_size = len(list(curve.keyframe_points))

        if target_size>current_size:
            print("Number of keyframes changed, adding keys to blender fcurve.")
            add_key(obj_name, fullname, target_size-current_size)
        if target_size<current_size:
            print("Number of keyframes changed, removing keys from blender fcurve.")
            remove_key(obj_name, fullname, current_size-target_size)

        ## problem here

        attr_names = C.Attributes_Name
        
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
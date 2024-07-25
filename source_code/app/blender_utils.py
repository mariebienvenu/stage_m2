
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
        print(o.name, o.location, o.animation_data)

def list_scenes():
    # print all scene names in a list
    print(bpy.data.scenes.keys())


def get_filename():
    return bpy.data.filepath.replace('\\', '/')


def check_filename(filename):
    assert get_filename()==filename, f"Wrong file opened. Expected {filename} and found {get_filename()}."


def is_match(curve_name:str, other_curve_name:str):
    first = curve_name.replace('"', "'")
    second = other_curve_name.replace('"', "'")
    return first==second

def do_stuff(obj_name='RIG-rex'): 
    # for debug
    for o in bpy.data.objects:
        anim_data, action, curves = False, False, False
        if o.name == obj_name:
            anim_data = getattr(o, "animation_data", False)
            if anim_data is not False:
                action = getattr(anim_data, "action", False)
                if action is not False:
                    curves = getattr(action, 'fcurves', False)
            print(o.name, anim_data, action, curves)
            print(o.animation_data)
            print(o.animation_data.action)
            print(anim_data.action)
            print(o.animation_data.action.fcurves)

    
def get_animation(obj_name):
    animation = Animation()

    obj = None
    for o in bpy.data.objects:
        if o.name == obj_name:
            obj = o
            break # TODO sometimes the algo breaks because there is several obj_name, and maybe the last one has no fcurves (action=None...) ; don't know why. caused with RIG-rex.
    if obj is None:
        return animation
    
    debug = obj.animation_data.action
    
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
            fullname=f"{curve.data_path} {['X','Y','Z'][curve.array_index]}".replace('"', "'"),
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
        if is_match(curve_name,fullname):
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
        if is_match(curve_name,fullname):
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
        target_curve = None
        for anim_curve in animation:
            if is_match(anim_curve.fullname, fullname): target_curve=anim_curve
        if target_curve is None:
            debug = 0
            raise NotImplementedError(f"euh pas trouvé {fullname} dans {animation}")

        target_size = len(target_curve)
        current_size = len(list(curve.keyframe_points))

        if target_size>current_size:
            print(f"Number of keyframes changed, adding {target_size-current_size} keys to blender fcurve.")
            add_key(obj_name, fullname, target_size-current_size)
        if target_size<current_size:
            print(f"Number of keyframes changed, removing {current_size-target_size} keys from blender fcurve.")
            remove_key(obj_name, fullname, current_size-target_size)

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
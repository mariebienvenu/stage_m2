from typing import Protocol, List
from enum import Enum

from types.curve_types import AnimationCurve

class RotationOrder(Enum):
    "Specifies the rotation order of an object."
    XYZ = 0
    XZY = 1
    YXZ = 2
    YZX = 3
    ZXY = 4
    ZYX = 5


class Scene(Protocol):
    """All objects' rotations are assumed to be modeled by Euler angles."""

    def get_all_animated_objects(self) -> List[str] : """Lists all the controllers in the scene which hold animation curves."""
    def get_animated_attributes(self, object_name:str) -> List[str] : """Lists all the animated attributes of an animation controller. An attribute is considered animated if it has at least two keys."""
    def get_rotation_order(self, object_name:str) -> RotationOrder : ...
    def get_animation_curve(self, object_name:str, attribute:str) -> AnimationCurve : ...
    def get_time_range(self) -> tuple[int, int] : """The scene's time range as defined by the time slider."""
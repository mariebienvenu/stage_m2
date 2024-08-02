
from typing import Protocol
from enum import Enum

## DONE : what about objects in the 3D scene ? -> see scene_types

class ValueType(Enum):
    """Specifies what kind of attribute the curve models."""
    POSITION = 0
    ORIENTATION = 1
    SCALE = 2
    OTHER = 3 # Useful for "Visibility" for instance, or IK/FK switch...

class Axis(Enum):
    """Specifies what axis the curve belongs to, if any."""
    X = 0
    Y = 1
    Z = 2
    NEITHER = 3 # Useful for "Visibility" for instance, or IK/FK switch...

class TangentState(Enum):
    """Specifies whether the two handles of a tangent are linked or not."""
    ALIGNED = 0
    BROKEN = 1

class KeyframeTypeToAdd(Enum):
    """Specifies at what stage a keyframe is, in terms of the classic animation pipeline : Blocking/Posing, Spline/Polish"""
    BLOCKING = 0 # Constant interpolation
    SPLINE = 1 # Bézier interpolation


class AnimationCurve(Protocol):
    """Link between a given DCC and the "Editing animation through gestures" project."""

    @property
    def keyframe_count(self) -> int : """The number of keys in the curve."""
    
    @property
    def time_range(self) -> tuple[float, float] : """Start, stop (in frames)"""
    
    @property
    def data_type(self) -> ValueType : """The kind of attribute the curve represents : orientation, position, scale, etc..."""
    
    @property
    def axis(self) -> Axis : """The axis the curve's data belongs to : X, Y, Z, or neither"""
    
    @property
    def color(self) -> tuple[int, int, int] | tuple[int, int, int, float] | str : """The color of the curve, used for visualisation purposes ; RGB code, RGBA code, CSS color name, or hex code."""
    
    @property
    def name(self) -> str : """The name of the curve, used for debug and lisibility ; example : "Ball_1, Location X". """

    @property
    def id(self) : """Unique identifier of the curve. Used for sanity checks."""

    @property
    def key_type_to_add(self) -> KeyframeTypeToAdd : """The type of interpolation chosen when adding a new key to the curve"""

    ## IMPORTANT
    def evaluate(time:float) -> float : """Evaluates the curve at given time (in frames)."""
    def add_key() -> None : """Adds a key of default value after the end of the curve using user's settings regarding interpolation and tangents."""
    def delete_key(index:int) -> None : """Deletes key number @index. All the keys later in time should have their index updated."""

    ## Getters
    def get_time(self, index:int) -> float : """X component of key number @index (in frames)."""
    def get_value(self, index:int) -> float : """Y component of key number @index."""

    def get_tangent_state(self, index:int) -> TangentState : """Whether the tangent of the key number @index is broken or not."""

    def get_left_tangent_time(self, index:int) -> float : """Left tangent handle's X component of key number @index (in frames)."""
    def get_left_tangent_value(self, index:int) -> float : """Left tangent handle's Y component of key number @index."""
    def get_right_tangent_time(self, index:int) -> float : """Right tangent handle's X component of key number @index (in frames)."""
    def get_right_tangent_value(self, index:int) -> float : """Right tangent handle's Y component of key number @index."""

    ## Setters
    def set_time(self, index:int, time:float) -> None : ...
    def set_value(self, index:int, value:float) -> None : ...
    def set_tangent_state(self, index:int, state:TangentState) -> None : ...
    def set_left_tangent_time(self, index:int, time:float) -> None : ...
    def set_left_tangent_value(self, index:int, value:float) -> None : ...
    def set_right_tangent_time(self, index:int, time:float) -> None : ...
    def set_right_tangent_value(self, index:int, value:float) -> None : ...

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

class KeyType(Enum):
    """Specifies the behaviour of the key when moving neigbour keys."""
    KEYFRAME = 0 ; """Fixed key. Extremes, Moving Holds, and other special cases should be seen as fixed keys."""
    BREAKDOWN = 1 ; """Moves when moving neighbour keys."""

class TangentState(Enum):
    """Specifies whether the two handles of a tangent are linked or not."""
    ALIGNED = 0
    BROKEN = 1

class TangentBehaviour(Enum):
    """Specifies how the user defined the tangents."""
    USER_DEFINED = 0 ; """The user defined custom values for the tangent."""
    LINEAR = 1 ; """The tangent is set to produce linear interpolation (usually combined with TangentType.BROKEN)."""
    FLAT = 2 ; """The tangent is perfectly flat."""
    AUTO = 3 ; """The tangent is calculated by the DCC (usually combined with TangentState.ALIGNED)."""

# NEW : replaced Interpolation
class AnimationStage(Enum):
    """Specifies at what stage a keyframe is, in terms of the classic animation pipeline : Blocking/Posing, Spline/Polish"""
    BLOCKING = 0
    SPLINE = 1


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
    def id() : """Unique identifier of the curve. Used for sanity checks."""

    ## IMPORTANT
    def evaluate(time:float) -> float : """Evaluates the curve at given time (in frames)."""
    def add_key() -> None : """Adds a key of default value after the end of the curve using user's settings regarding interpolation and tangents."""
    def delete_key(index:int) -> None : """Deletes key number @index. All the keys later in time should have their index updated."""

    ## Getters
    def get_time(self, index:int) -> float : """X component of key number @index (in frames)."""
    def get_value(self, index:int) -> float : """Y component of key number @index."""
    def get_keytype(self, index:int) -> KeyType : """Whether the key number @index is a Keyframe or a Breakdown."""
    def get_animation_stage(self, index:int) -> AnimationStage : """Whether the key number @index is at the Blocking or Spline stage."""

    def get_tangent_state(self, index:int) -> TangentState : """Whether the tangent of the key number @index is broken or not."""

    def get_left_tangent_x(self, index:int) -> float : """Left tangent vector's X component of key number @index (in frames)."""
    def get_left_tangent_y(self, index:int) -> float : """Left tangent vector's Y component of key number @index."""
    def get_right_tangent_x(self, index:int) -> float : """Right tangent vector's X component of key number @index (in frames)."""
    def get_right_tangent_y(self, index:int) -> float : """Right tangent vector's Y component of key number @index."""

    def get_left_tangent_behaviour(self, index:int) -> TangentBehaviour : """Left tangent's behaviour of key number @index."""
    def get_right_tangent_behaviour(self, index:int) -> TangentBehaviour : """Right tangent's behaviour of key number @index."""

    ## Setters
    def set_time(self, index:int, time:float) -> None : ...
    def set_value(self, index:int, value:float) -> None : ...
    def set_tangent_state(self, index:int, state:TangentState) -> None : ...
    def set_left_tangent_x(self, index:int, x:float) -> None : ...
    def set_left_tangent_y(self, index:int, y:float) -> None : ...
    def set_right_tangent_x(self, index:int, x:float) -> None : ...
    def set_right_tangent_y(self, index:int, y:float) -> None : ...
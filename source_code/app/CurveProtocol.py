
from typing import Protocol
from enum import Enum

## TODO : what about objects in the 3D scene ? like a get_animation(obj_name) -> List[AnimationCurve]

class ValueType(Enum):
    POSITION = 0
    ORIENTATION = 1
    SCALE = 2
    OTHER = 3 # Useful for "Visibility" for instance, or IK/FK switch...

class Axis(Enum):
    X = 0
    Y = 1
    Z = 2
    NEITHER = 3 # Useful for "Visibility" for instance, or IK/FK switch...

class KeyType(Enum):
    KEYFRAME = 0
    BREAKDOWN = 1
    OTHER = 2 # Extreme, Moving Hold, etc

class TangentState(Enum):
    ALIGNED = 0
    BROKEN = 1

class TangentBehaviour(Enum):
    USER_DEFINED = 0 # The user defined custom values for the tangent
    LINEAR = 1 # The tangent is set to produce linear interpolation (usually combined with TangentType.BROKEN)
    FLAT = 2 # The tangent is perfectly flat
    AUTO = 3 # The tangent is calculated by the DCC (usually combined with TangentType.ALIGNED)

class Interpolation(Enum):
    CONSTANT = 0 # Equivalent of "step"
    CONSTANT_NEXT = 1 # Step but with next keyframe's value
    BEZIER = 2
    OTHER = 3 # Useful for other types of interpolation such as "Cubic" for instance


class AnimationCurve(Protocol):

    def __len__(self) -> int : ... # Returns the number of keyframes
    
    @property
    def time_range(self) -> tuple[float, float] : ... # Start, stop (in frames)
    
    @property
    def data_type(self) -> ValueType : ... # Whether the curve represents orientation, position, scale, other...
    
    @property
    def axis(self) -> Axis : ... # Whether the curve represents data along X, Y, Z, or neither
    
    @property
    def color(self) : ... # For visualisation purposes ; RGB or RGBA or "classic" color name
    
    @property
    def name(self) -> str : ... # For debug and lisibility ; example : "{object short name} Location X"

    @property
    def id() : ... # Will be used for sanity checks

    ## IMPORTANT
    def evaluate(time:float) -> float : ... # Evaluates the curve at given time (in frames)
    def add_key() -> int : ... # Adds a key and returns its index
    def delete_key(index:int) -> None : ...

    # Getters
    def get_id(self, index:int) : ... # Will be used for sanity checks
    def get_time(self, index:int) -> float : ...
    def get_value(self, index:int) -> float : ...
    def get_keytype(self, index:int) -> KeyType : ...
    def get_tangent_state(self, index:int) -> TangentState : ... # Whether the tangent is broken or not

    def get_left_tangent_xy(self, index:int) -> tuple[float, float] : ... # Position relative to the keyframe
    def get_right_tangent_xy(self, index:int) -> tuple[float, float] : ...

    def get_left_tangent_behaviour(self, index:int) -> TangentBehaviour : ...
    def get_right_tangent_behaviour(self, index:int) -> TangentBehaviour : ...

    def get_left_interpolation(self, index:int) -> Interpolation : ... # How the leftward interval is interpolated
    def get_right_interpolation(self, index:int) -> Interpolation : ... # How the rightward interval is interpolated

    # Setters
    def set_time(self, index:int, time:float) -> None : ...
    def set_value(self, index:int, value:float) -> None : ...
    def set_keytype(self, index:int, keytype:KeyType) -> None : ...
    def set_tangent_state(self, index:int, state:TangentState) -> None : ...
    def set_left_tangent_xy(self, index:int, xy:tuple[float, float]) -> None : ...
    def set_right_tangent_xy(self, index:int, xy:tuple[float, float]) -> None : ...
    def set_left_tangent_behaviour(self, index:int, behaviour:TangentBehaviour) -> None : ...
    def set_right_tangent_behaviour(self, index:int, behaviour:TangentBehaviour) -> None : ...
    def set_left_interpolation(self, index:int, interpolation:Interpolation) -> None : ... # should not impact extrapolation 
    def set_right_interpolation(self, index:int, interpolation:Interpolation) -> None : ... # should not impact extrapolation 
    
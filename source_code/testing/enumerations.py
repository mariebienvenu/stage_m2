from enum import Enum

class Easing_Mode(Enum):
    AUTO = 0 # most used
    EASE_IN = 1
    EASE_OUT = 2
    EASE_IN_OUT = 3

mode = Easing_Mode.AUTO
print(mode, type(mode), mode.value, mode.name, mode._name_)

mode_name = 'AUTO'
res = eval(f"Easing_Mode.{mode_name}") # do NOT do that !
print(mode_name, Easing_Mode(0), res, type(res), res.value)

res2 = Easing_Mode[mode_name] # prefered syntax to call by name !
print(res2)

print(len(list(Easing_Mode)))

print(Easing_Mode(0)) # we can also call by value
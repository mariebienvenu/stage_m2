
import plotly.express as px
from webcolors import name_to_rgb

class Color:
    current = -1
    colors = [tuple(int(hex[i:i+2], 16) for i in (1, 3, 5)) for hex in list(px.colors.qualitative.Plotly)]

    @classmethod
    def next(cls):
        cls.current += 1
        cls.current = cls.current % len(cls.colors)
        return cls.colors[cls.current]
    
    @classmethod
    def reset(cls):
        cls.current = -1

    @staticmethod
    def to_string(color:tuple[int, float]|str, opacity:float=1.):
        if opacity==1 and type(color)==str: return color
        if type(color)==str:
            if 'rgba' in color:
                color = tuple(color[5:-1].split(','))
            elif 'rgb' in color:
                color = tuple(color[4:-1].split(','))
            else:
                color = tuple(name_to_rgb(color))
            color = tuple([float(e) for e in color])
        if len(color)==3:
            r,g,b = color
            return f'rgba({r},{g},{b},{opacity})'
        elif len(color)==4:
            r,g,b,a = color
            return f'rgba({r},{g},{b},{a*opacity})'
        raise AssertionError(f"Provided color is not in correct format. Expected tuple of len 3 or 4, or string ; got {type(color).__name__}")

import plotly.express as px

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
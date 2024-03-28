
import plotly.express as px

class Color:
    current = -1
    colors = [tuple(int(hex[i:i+2], 16) for i in (1, 3, 5)) for hex in list(px.colors.qualitative.Plotly)]

    @staticmethod
    def next():
        Color.current += 1
        Color.current = Color.current % len(Color.colors)
        return Color.colors[Color.current]
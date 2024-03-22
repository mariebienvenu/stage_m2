import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import numpy as np

def add_curve(y, color, x = None, name = "", fig = None, row = None, col = None):
    Y = np.array(y)
    X = np.array(x) if x is not None else np.array(list(range(Y.size)))
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=Y, name=name, line_color=color), row=row, col=col)
    return fig


def add_aplat(under, above, color, x = None, name = "", fig = None, row = None, col = None):

    UNDER = np.array(under)
    ABOVE = np.array(above)
    X = np.array(x) if x is not None else np.array(list(range(ABOVE.size)))
    if fig is None:
        fig = go.Figure()

    fig.add_trace(go.Scatter(
        name=f'{name} Upper Bound',
        x=X,
        y=ABOVE,
        mode='lines',
        line_width=0,
        showlegend=False
    ), row=row, col=col)
    fig.add_trace(go.Scatter(
        name=f'{name} Lower Bound',
        x=X,
        y=UNDER,
        line=dict(width=0),
        mode='lines',
        fill='tonexty',
        fillcolor=color,
        showlegend=False
    ), row=row, col=col)

    return fig


def magnitude_angle(
        magnitude_means,
        magnitude_stds,
        angle_means,
        angle_stds,
        oflow_len,
        name,
        path,
        colors = [tuple(int(hex[i:i+2], 16) for i in (1, 3, 5)) for hex in list(px.colors.qualitative.Plotly)],
        doShow=False
):
        
    fig = make_subplots(rows=2, cols=1)

    add_curve(magnitude_means, f'rgb{colors[0]}', name='Magnitude', fig=fig, row=1, col=1)
    add_curve(angle_means, f'rgb{colors[1]}', name='Angle', fig=fig, row=2, col=1)

    add_aplat(magnitude_means-magnitude_stds, magnitude_means+magnitude_stds, f'rgba({colors[0][0]},{colors[0][1]},{colors[0][2]},0.3)', name='Magnitude', fig=fig, row=1, col=1)
    add_aplat(angle_means-angle_stds, angle_means+angle_stds, f'rgba({colors[1][0]},{colors[1][1]},{colors[1][2]},0.3)', name='Angle', fig=fig, row=2, col=1)

    fig.update_layout(title=f'Optical flow  - {name}')

    fig.update_xaxes(title_text="Frame number", row=2, col=1)
    fig.update_yaxes(title_text="Magnitude in pixels", row=1, col=1)
    fig.update_yaxes(title_text="Angle in degrees", row=2, col=1)

    fig.write_html(path+f"/{name}_oflow.html")
    if doShow: fig.show()
    return fig
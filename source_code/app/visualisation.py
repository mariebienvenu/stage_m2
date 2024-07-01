

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from app.color import Color


def add_curve(y, color=None, opacity=None, x=None, name="", fig=None, row=None, col=None, style="lines", legend=None):
    if opacity is None : opacity = 1.
    color = Color.to_string(color, opacity) if color is not None else Color.to_string(Color.next(), opacity)
    Y = np.array(y)
    X = np.array(x) if x is not None else np.array(list(range(Y.size)))
    if fig is None: fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=Y, name=name, line_color=color, mode=style, showlegend=legend), row=row, col=col)
    return fig


def add_aplat(under, above, color=None, opacity=None, x=None, name="", fig=None, row=None, col=None):
    if opacity is None : opacity = 1.
    color = Color.to_string(color, opacity) if color is not None else Color.to_string(Color.next(), opacity)
    
    UNDER = np.array(under)
    ABOVE = np.array(above)
    X = np.array(x) if x is not None else np.array(list(range(ABOVE.size)))
    if fig is None: fig = go.Figure()

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
        frame_times,
        magnitude_means,
        magnitude_stds,
        angle_means,
        angle_stds,
        colors = None,
        fig=None,
        doShow=False,
        rows=[1,2],
        cols=[1,1]
):
    if fig is None: fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Magnitude", "Angle"], vertical_spacing=0.1) # cannot put it as default arg because it is evaluated at launch time then used for everyone... since is pointer -> bad
    
    color1, color2 = colors if colors is not None else (Color.next(), Color.next())
    add_curve(magnitude_means, color1, x=frame_times, name='amplitude=f(t) - Amplitude du flux', fig=fig, row=rows[0], col=cols[0])
    add_curve(angle_means, color2, x=frame_times, name='angle=f(t) - Angle du flux', fig=fig, row=rows[1], col=cols[1])

    add_aplat(magnitude_means-magnitude_stds, magnitude_means+magnitude_stds, color1, opacity=0.3, x=frame_times, name='Magnitude', fig=fig, row=rows[0], col=cols[0])
    add_aplat(angle_means-angle_stds, angle_means+angle_stds, color2, opacity=0.3, x=frame_times, name='Angle', fig=fig, row=rows[1], col=cols[1])

    if doShow: fig.show()
    return fig


def add_pairings(y1:np.ndarray, y2:np.ndarray, pairs:list[list[int]], x1:list|np.ndarray=None, x2:list|np.ndarray=None, color=None, opacity=0.3, fig=None, row=None, col=None, doShow=False):
    color = Color.to_string(color, opacity) if color is not None else Color.to_string((100,100,100), opacity)
    if fig is None: fig = go.Figure()
    if x1 is None: x1 = list(range(y1.size))
    if x2 is None: x2 = list(range(y2.size))
    x1, x2 = np.array(x1), np.array(x2)
    for pair in pairs:
        i,j = pair
        fig.add_trace(go.Scatter(x=[x1[i], x2[j]], y=[y1[i], y2[j]], showlegend=False, line_color=color, mode="lines"), row=row, col=col)
    if doShow: fig.show()
    return fig


def add_heatmap(df:pd.DataFrame, min=None, max=None, is_correlation=False, fig=None, row=None, col=None, doShow=False):
    if fig is None: fig = go.Figure()
    if type(df) is np.ndarray: df = pd.DataFrame(df) # new
    colorscale = 'RdBu' if is_correlation else 'Plasma'
    fig.add_trace(go.Heatmap(z=df, x=df.columns, y=df.index, colorscale=colorscale, zmax=max, zmin=min), row=row, col=col)
    if doShow: fig.show()
    return fig


def add_circle(center:tuple, radius:float=2, color=None, name=None, fig=None, row=None, col=None, doShow=False):
    if fig is None: fig = go.Figure()
    color = Color.to_string(color) if color is not None else Color.to_string(Color.next())
    x,y = center
    fig.add_shape(
        type="circle",
        x0=x-radius, y0=y-radius, x1=x+radius, y1=y+radius,
        line_color=color,
        name=name,
        col=col,
        row=row
    )
    if doShow: fig.show()
    return fig
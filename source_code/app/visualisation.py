
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from app.Color import Color


def add_curve(y, color=None, x = None, name = "", fig = None, row = None, col = None, style="lines"):
    color = color if color is not None else f'rgb{Color.next()}'
    Y = np.array(y)
    X = np.array(x) if x is not None else np.array(list(range(Y.size)))
    if fig is None:
        fig = go.Figure()
    fig.add_trace(go.Scatter(x=X, y=Y, name=name, line_color=color, mode=style), row=row, col=col)
    return fig


def add_aplat(under, above, color=None, x = None, name = "", fig = None, row = None, col = None):
    color = color if color is not None else f'rgb{Color.next()}'

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
    fig = fig if fig is not None else make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Magnitude", "Angle"], vertical_spacing=0.1) # cannot put it as default arg because it is evaluated at launch time then used for everyone... since is pointer -> bad
    
    colors = colors if colors is not None else (Color.next(), Color.next())
    add_curve(magnitude_means, f'rgb{colors[0]}', x=frame_times, name='amplitude=f(t) - Amplitude du flux', fig=fig, row=rows[0], col=cols[0])
    add_curve(angle_means, f'rgb{colors[1]}', x=frame_times, name='angle=f(t) - Angle du flux', fig=fig, row=rows[1], col=cols[1])

    add_aplat(magnitude_means-magnitude_stds, magnitude_means+magnitude_stds, f'rgba({colors[0][0]},{colors[0][1]},{colors[0][2]},0.3)', x=frame_times, name='Magnitude', fig=fig, row=rows[0], col=cols[0])
    add_aplat(angle_means-angle_stds, angle_means+angle_stds, f'rgba({colors[1][0]},{colors[1][1]},{colors[1][2]},0.3)', x=frame_times, name='Angle', fig=fig, row=rows[1], col=cols[1])

    if doShow: fig.show()
    return fig


def add_pairings(y1, y2, pairs, x1=None, x2=None, color=None, fig=None, row=None, col=None, doShow=False):
    fig = fig if fig is not None else go.Figure()
    color = color if color is not None else 'rgba(100,100,100, 0.3)'
    x1 = np.array(x1) if x1 is not None else np.array(list(range(y1.size)))
    x2 = np.array(x2) if x2 is not None else np.array(list(range(y2.size)))
    for pair in pairs:
        i,j = pair
        fig.add_trace(go.Scatter(x=[x1[i], x2[j]], y=[y1[i], y2[j]], showlegend=False, line_color=color, mode="lines"), row=row, col=col)
    if doShow: fig.show()
    return fig


def add_heatmap(df:pd.DataFrame, min=None, max=None, is_correlation=False, fig=None, row=None, col=None, doShow=False):
    fig = fig if fig is not None else go.Figure()
    colorscale = 'RdBu' if is_correlation else 'Plasma'
    fig.add_trace(go.Heatmap(z=df, x=df.columns, y=df.index, colorscale=colorscale, zmax=max, zmin=min), row=row, col=col)
    if doShow: fig.show()
    return fig
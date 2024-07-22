
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.color import Color
from app.curve import Curve
from app.animation import Animation
from app.semantic import SemanticRetiming

DO_SHOW = True

## let's make a simple bouncing ball curve
times = np.array([0, 15, 30, 45, 60])
values = np.array([0, 0, 0, 0, 0])
coordinates = np.vstack((times, values)).T

width = 1
left_handle_x = np.expand_dims(times-width, axis=1)
right_handle_x = np.expand_dims(times+width, axis=1)
left_handle_y = np.expand_dims(np.array([0, 3, 3, 3, 3]), axis=1)
right_handle_y = np.expand_dims(np.array([3, 3, 3, 3, 0]), axis=1)

curve = Curve(
    coordinates,
    tangent_left_handle_x=left_handle_x,
    tangent_left_handle_y=left_handle_y,
    tangent_right_handle_x=right_handle_x,
    tangent_right_handle_y=right_handle_y,
    fullname="Location Z"
)

## Now let's imagine we got some retiming matches with slight errors
matches = np.array([
    [0, 0],
    [14, 30], #15
    [35, 45], #30
    [42, 48], #45 
    [57, 93] #60l
])

channels = ["Location Z"]
animation = Animation([curve])

## Okay, let's go !
semantic = SemanticRetiming(animation, channels, matches)
new_animation = semantic.process()

## Let's make sure the computation worked
snapped = times
print(f"Expected:{snapped}, got {semantic.snapped_times_reference}")
scaling = [1, 2, 1, 0.2, 3]
print(f"Expected:{scaling}, got {semantic.basic_left},{semantic.basic_right}")
print(f"Using basic weights:  {semantic.new_left},{semantic.new_right}")

## Illustration
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=["Original animation", "Retimed animation"])
animation.display(fig=fig, row=1,col=1)
new_animation[0].display(fig=fig, row=2,col=1, name="regular")

## What if we used no semantics ?
SemanticRetiming.REGULARIZATION_WEIGHT = 0
new_animation = semantic.process(force=True)
print(f"No regularization scenario: expected {scaling}, got {semantic.new_left}")
new_animation[0].display(fig=fig, row=2, col=1, name="no regularization", opacity=0.3)

## What if we used only semantics ?
SemanticRetiming.REGULARIZATION_WEIGHT = 100
new_animation = semantic.process(force=True)
print(f"Extreme regularization scenario: {semantic.new_left}, {semantic.new_right}")
new_animation[0].display(fig=fig, row=2, col=1, name="no matches", opacity=0.3)

if DO_SHOW: fig.show()
SemanticRetiming.reset_weights()

## let's play with the amount of neighbour impact (symmetries)
SemanticRetiming.REGULARIZATION_WEIGHT = 100

# Case of : no neigbours
SemanticRetiming.NEIGHBOURS_WEIGHT = 0
SemanticRetiming.ALIGNMENT_WEIGHT = 1
SemanticRetiming.BROKEN_WEIGHT = 1
new_animation = semantic.process(force=True)
print(f"Extreme regularization + no neighbours scenario: expected [1.5, 1.5, 0.6, 1.6, 2], got {semantic.new_left}, {semantic.new_right}")

# Case of : full neighbours
SemanticRetiming.NEIGHBOURS_WEIGHT = 1
SemanticRetiming.ALIGNMENT_WEIGHT = 0
SemanticRetiming.BROKEN_WEIGHT = 0
new_animation = semantic.process(force=True)
print(f"Extreme regularization + full neighbours scenario: got {semantic.new_left}") # Same as no regularization ! because neighbour continuity is already granted...

# Case of : everything
SemanticRetiming.NEIGHBOURS_WEIGHT = 1
SemanticRetiming.ALIGNMENT_WEIGHT = 1
SemanticRetiming.BROKEN_WEIGHT = 1
new_animation = semantic.process(force=True)
print(f"Extreme regularization + max everything scenario: expected {np.average(scaling)}, got {semantic.new_left}") # Uniform scaling


## Other scenario : aligned keys (no contact, smooth curve)
SemanticRetiming.reset_weights()
Color.reset()

times = np.array([0, 15, 30, 45, 60])
values = np.array([0, 0, 0, 0, 0])
coordinates = np.vstack((times, values)).T

width = 1
left_handle_x = np.expand_dims(times-width, axis=1)
right_handle_x = np.expand_dims(times+width, axis=1)
left_handle_y = np.expand_dims(np.array([-3, 3, -3, 3, -3]), axis=1)
right_handle_y = np.expand_dims(np.array([3, -3, 3, -3, 3]), axis=1)

curve = Curve(
    coordinates,
    tangent_left_handle_x=left_handle_x,
    tangent_left_handle_y=left_handle_y,
    tangent_right_handle_x=right_handle_x,
    tangent_right_handle_y=right_handle_y,
    fullname="Location Z",
    color = Color.next()
)

matches = np.array([
    [0, 0],
    [14, 30], #15
    [35, 45], #30
    [42, 48], #45 
    [57, 93] #60l
])

channels = ["Location Z"]
animation = Animation([curve])

## Okay, let's go !
semantic = SemanticRetiming(animation, channels, matches)
new_animation = semantic.process()

## Illustration
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, subplot_titles=["Original animation", "Retimed animation"])
animation.display(fig=fig, row=1,col=1)
new_animation.display(fig=fig, row=2,col=1)
if DO_SHOW: fig.show()

## What if we varied the level of regularization ?
weights = [0, 1, 5, 20]
fig = make_subplots(rows=1, cols=len(weights), shared_xaxes=True, shared_yaxes=True, subplot_titles=[f"Weight={w}" for w in weights])
for i,weight in enumerate(weights):
    SemanticRetiming.REGULARIZATION_WEIGHT = weight
    new_animation = semantic.process(force=True)
    new_animation[0].display(fig=fig, row=1, col=i+1)
if DO_SHOW: fig.show()

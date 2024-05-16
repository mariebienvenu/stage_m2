

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

import app.visualisation as vis

np.random.seed(5)

## First, let's create some data

t0 = np.linspace(1, 10, 10)
x0 = np.array([6, 8, 4, 2, 5, 3, 9, 1, 0, 7])
x1 = np.array([2, 3, 4, 6, 7, 8, 5, 1, 9, 0])

t2 = np.linspace(1, 10, 100)
x2 = 5*np.cos(t2 +0.2) #si on veut checker le cas où c'est pas synchrone
x3 = 7*np.cos(t2) - 2

t4 = np.linspace(1, 10, 100)
x4, x5 = np.random.random((100)), np.random.random((100))
x4[[10, 30, 72]] = [6, -5, 3]
x5[[10, 30, 72]] = [4, -8, 3]

data = [[t0, x0, x1], [t2, x2, x3], [t4, x4, x5]]

## Let's make some visualisations

fig = make_subplots(rows=2, cols=len(data))
for i, (t,x_1,x_2) in enumerate(data):
    vis.add_curve(y=x_1, x=t, fig=fig, row=1, col=1+i, style="lines")
    vis.add_curve(y=x_2, x=t, fig=fig, row=1, col=1+i, style="lines")
    vis.add_curve(y=x_2, x=x_1, fig=fig, row=2, col=1+i, style="markers")
fig.show()

fig2 = make_subplots(rows=2, cols=len(data))
for i, (t,x_1,x_2) in enumerate(data):
    vis.add_curve(y=t, x=x_1, fig=fig2, row=1, col=1+i, style="markers")
    vis.add_curve(y=t, x=x_2, fig=fig2, row=1, col=1+i, style="markers")
    vis.add_curve(y=x_2, x=x_1, fig=fig2, row=2, col=1+i, style="markers")
fig2.show()

## Now, let's create our matching function

t,x,y, = data[2]

x_index = np.argsort(x)
x_croissant = x[x_index]
y_index = np.argsort(y)
y_croissant = y[y_index]

print(f'OG X: {x} \n ARGSORT:{x_index} \n SORTED:{x_croissant}')
print(f'OG Y: {y} \n ARGSORT:{y_index} \n SORTED:{y_croissant}')

cost_matrix = np.array([[abs(t[idx]-t[idy]) for idy in y_index] for idx in x_index]) 

fig3 = vis.add_heatmap(pd.DataFrame(cost_matrix))
fig3.show()

def shortest_increasing_path(cost_matrix:np.ndarray):
    n,m = cost_matrix.shape
    cumulative_cost = np.ones((n+1,m+1))*np.inf
    cumulative_cost[0,0] = 0
    for i in range(n):
        for j in range(m):
            cost = cost_matrix[i,j]
            additionnal_cost = min(cumulative_cost[i+1,j], cumulative_cost[i,j+1], cumulative_cost[i, j])
            cumulative_cost[i+1,j+1] = cost + additionnal_cost
    score = cumulative_cost[n,m]
    reverse_path = [[n-1,m-1]]
    i,j = n,m
    while i>1 or j>1:
        current = cumulative_cost[i,j]
        precedent_cost = current - cost_matrix[i-1, j-1]
        if cumulative_cost[i-1, j-1] == precedent_cost:
            i -= 1
            j -= 1
        elif cumulative_cost[i, j-1] == precedent_cost:
            j -= 1
        elif cumulative_cost[i-1, j] == precedent_cost:
            i -= 1
        else:
            print(f"problem: {i,j,current,precedent_cost,cumulative_cost[i-1:i+1,j-1:j+1],cost_matrix[i-1, j-1]}")
        reverse_path.append([i-1,j-1])
    path = reverse_path[::-1]
    return path, score


def make_bijection(path, sorted_values1, sorted_values2):
    return (
        np.array([sorted_values1[i] for i,j in path]),
        np.array([sorted_values2[j] for i,j in path])
    )

path1, score1 = shortest_increasing_path(cost_matrix)
bij1 = make_bijection(path1, x_croissant, y_croissant)

fig4 = vis.add_curve(y=bij1[1], x=bij1[0], style="lines+markers")
fig4.show()
path2, score2 = shortest_increasing_path(cost_matrix[:,::-1])

print(f"Shortest increasing path has score: {score1} \n   PATH: {path1} \n   Bijection: {bij1}")
#print(f"Shortest decreasing path has score: {score2} \n   PATH: {path2}")

fig5 = vis.add_curve(y=t[x_index], x=x_croissant, style="lines")
vis.add_curve(y=t[y_index], x=y_croissant, fig=fig5, style="lines")
vis.add_pairings(y1=t[x_index], x1=x_croissant, y2=t[y_index], x2=y_croissant, pairs=path1, fig=fig5)
fig5.show()


## debug

first_pairs = [(x_croissant[i],y_croissant[j]) for i,j in path1[:10]]
print(first_pairs)

costs = [cost_matrix[i,j] for i,j in path1]
print(f"Score: {score1} Score:{sum(costs)}")
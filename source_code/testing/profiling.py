
import os
import cProfile
import re
from time import time

import numpy as np
import pandas as pd

temp_dir = 'C:/Users/Marie Bienvenu/stage_m2/afac/profiling/'

## Profiling test on basic stuff

def func(x=1):
    def aux(x):
        return x*2
    for _ in range(10):
        x+=1
        x = aux(x)
    return x

class A:

    def __init__(self):
        return

    def do(self):
        for x in range(10):
            func(x)

a = A()
cProfile.run('func()', temp_dir+'func_stats.txt')
cProfile.run('a.do()', temp_dir+'A_do_stats.txt')

## Profiling test on real stuff

import app.main as main
import app.dynamic_time_warping as DTW
import app.visualisation as vis

directory = "C:/Users/Marie Bienvenu/stage_m2/complete_scenes/empty/"
ref = "P1010261"
target = "P1010258"

exp_directory = directory+f'/{ref}_VS_{target}/'
main_obj = main.Main(directory, no_blender=True, verbose=2)
main_obj.config["video reference filename"] = ref
main_obj.config["video target filename"] = target
main_obj.load_videos() # to load correct videos
main_obj.process(force=True)

dtw:DTW.DynamicTimeWarping = main_obj.internals[0].dtw
cProfile.run('dtw.global_constraints()', filename=temp_dir+'global_constraints_stats.txt')
print("INTERRUPT")

## Trying to come up with a better distance function -> fail

cost_matrix = dtw.cost_matrix

def distances(costs:np.ndarray):
    n,m = costs.shape
    distances = np.ones((n+1,m+1))*np.inf
    distances[0,0] = 0
    for i in range(n):
        for j in range(m):
            cost = costs[i,j]
            additionnal_cost = min(distances[i+1,j], distances[i,j+1], distances[i, j])
            distances[i+1,j+1] = cost + additionnal_cost # c'est cette ligne la plus longue
    return distances

import scipy.sparse as sparse

def djikstra_distances(costs:np.ndarray): # nul
    n,m = costs.shape
    graph = np.zeros((n*m, n*m))
    for i in range(n):
        for j in range(m):
            index = i*m+j
            if i<n-1:
                graph[index+m] = costs[i+1,j]
            if j<m-1:
                graph[index+1] = costs[i,j+1]
            if i<n-1 and j<m-1:
                graph[index+1+m] = costs[i+1, j+1]
    distances : np.ndarray = sparse.csgraph.dijkstra(graph, directed=True, indices=[0]) ## too memory expensive ! 154 Gb
    return distances.reshape((n,m))


import matplotlib
import matplotlib.pyplot as plt

import matplotlib.animation as animation


def distances_avec_pile(costs:np.ndarray):
    n,m = costs.shape
    distances = np.ones((n,m))*np.inf
    distances[0,0] = 0
    to_do = np.zeros((n,m), dtype=bool)
    to_do[0,0] = True
    masked = np.ravel(np.where(to_do==True, distances, np.inf))
    indexes = np.array([[i*m+j for j in range(m)] for i in range(n)])

    #debug
    #vis.add_heatmap(pd.DataFrame(costs), doShow=True)
    '''fig = plt.figure(figsize=(8,8))
    im = plt.imshow(to_do.astype(int)[::-1,:], interpolation='none', aspect='auto', vmin=0, vmax=1)
    fps = 30'''

    i_ = []
    j_ = []
    index_ = []
    index__ = []

    def update():
        index = np.argmin(masked) # most time consuming, even when handling conversion to flat ourselves
        i,j = index//m, index%m
        i_.append((i+1)/n)
        j_.append((j+1)/m)
        index_.append((index+1)/masked.size)
        index__.append(index)
        to_do[i,j]=False
        masked[i*m+j] = np.inf
        if i==n-1 and j==m-1:
            print("finished")
            '''vis.add_heatmap(pd.DataFrame(to_do), doShow=True)'''
            return True
        distance = distances[i,j]
        for (new_i, new_j) in zip([i,i+1, i+1], [j+1, j, j+1]):
            if new_i<n and new_j<m:
                challenger = distance+costs[new_i, new_j]
                if challenger < distances[new_i,new_j]:
                    distances[new_i,new_j] = challenger
                    to_do[new_i,new_j] = True
                    masked[new_i*m+new_j] = challenger
        return False
    
    def update2():
        candidate_indexes = indexes[to_do]
        candidate_costs = distances[to_do]
        temp_index = np.argmin(candidate_costs)
        index = candidate_indexes[temp_index]
        i,j = index//m, index%m
        to_do[i,j]=False
        if i==n-1 and j==m-1:
            print("finished")
            """vis.add_heatmap(pd.DataFrame(to_do), doShow=True)"""
            return True
        distance = distances[i,j]
        for (new_i, new_j) in zip([i,i+1, i+1], [j+1, j, j+1]):
            if new_i<n and new_j<m:
                challenger = distance+costs[new_i, new_j]
                if challenger < distances[new_i,new_j]:
                    distances[new_i,new_j] = challenger
                    to_do[new_i,new_j] = True
        return False

    '''
    def animate_func(i):
        for _ in range(100):
            update2()
        im.set_array(to_do[::-1,:])
        return [im]

    anim = animation.FuncAnimation(
        fig, 
        animate_func,
        frames = fps*15,
        interval = 1000 / fps, # in ms
    )

    plt.show()'''

    while np.any(to_do):
        b = update2()
        if b:
            print("greedy stop")
            '''plt.imshow(to_do.astype(int)[::-1,:], interpolation='none', aspect='auto', vmin=0, vmax=1)
            plt.show()'''
            return distances

    '''print(max(index__), masked.size)
    fign = vis.add_curve(y=i_)
    vis.add_curve(y=j_, fig=fign)
    vis.add_curve(y=index_, fig=fign)
    fign.show()'''
    return distances

cProfile.run('distances_avec_pile(cost_matrix)', filename=temp_dir+'pile_stats.txt')

t0 = time()
for _ in range(100):
    d1 = distances(cost_matrix)
t1 = time()
d2 = distances_avec_pile(cost_matrix)
t2 = time()

print((t1-t0)/100)
print(t2-t1)

vis.add_heatmap(pd.DataFrame(d1), doShow=True)
vis.add_heatmap(pd.DataFrame(d2), doShow=True)
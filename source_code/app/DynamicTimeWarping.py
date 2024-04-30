
import numpy as np

from app.Curve import Curve



class DynamicTimeWarping: ## TODO : make dedicated test file using content of maths.py


    def __init__(self, curve1:Curve, curve2:Curve):
        self.curve1 = curve1
        self.curve2 = curve2
        self.times1, self.times2 = curve1.get_times(), curve2.get_times()
        self.compute()

        self.bijection = (
            np.array([self.times1[i] for i,j in self.pairings[::-1]]),
            np.array([self.times2[j] for i,j in self.pairings[::-1]])
        )
    

    def compute(self):
        self.values1, self.values2 = np.ravel(self.curve1.get_values()), np.ravel(self.curve2.get_values())
        n,m = self.values1.size, self.values2.size
        self.cost_matrix = np.array([[abs(self.values1[i] - self.values2[j]) for j in range(m)] for i in range(n)]) # distance
        self.DTW = np.ones((n+1,m+1))*np.inf
        self.DTW[0,0] = 0
        for i in range(n):
            for j in range(m):
                cost = self.cost_matrix[i,j]
                additionnal_cost = min(self.DTW[i+1,j], self.DTW[i,j+1], self.DTW[i, j])
                self.DTW[i+1,j+1] = cost + additionnal_cost
        self.score = self.DTW[n,m]
        self.pairings = [[n-1,m-1]]
        i,j = n,m
        while i>1 or j>1:
            current = self.DTW[i,j]
            if self.DTW[i-1, j-1] <= current:
                i -= 1
                j -= 1
            elif self.DTW[i, j-1] <= current:
                j -= 1
            elif self.DTW[i-1, j] <= current:
                i -= 1
            self.pairings.append([i-1,j-1])
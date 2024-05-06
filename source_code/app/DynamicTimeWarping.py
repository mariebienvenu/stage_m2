
import numpy as np

from app.Curve import Curve


class DynamicTimeWarping:


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


    def local_constraints(self, window_size=10):
        range_x = len(self.curve1)
        range_y = len(self.curve2)
        N = self.bijection[0].size
        local_constraints = np.zeros((N))
        for i in range(1, N-1):
            ix,iy = np.array(self.pairings)[::-1][i] ## TODO make pairings an int array from start ? aller voir vis.add_pairings
            center_cost = self.cost_matrix[ix, iy] # best cost (globally)
            w_size = min(window_size, ix, iy, range_x-ix, range_y-iy)
            upper_costs = self.cost_matrix[ix+1:ix+w_size, iy] + self.cost_matrix[ix, iy+1:iy+w_size]
            lower_costs = self.cost_matrix[ix-w_size:ix, iy] + self.cost_matrix[ix, iy-w_size:iy]
            alternative_costs = np.concatenate((upper_costs, lower_costs))
            minimal_additionnal_cost = np.min(alternative_costs) - center_cost if alternative_costs.size>0 else 0
            local_constraints[i] = max(0,minimal_additionnal_cost)
        return local_constraints
    

    def global_constraints(self, debug=False):
        N = self.bijection[0].size
        n,m = self.values1.size, self.values2.size

        if debug: DTWs = np.zeros((N, n+1, m+1))

        global_constraints = np.zeros((N))
        cost_matrix = np.copy(self.cost_matrix)

        for index in range(1, N-1):

            previous_ix, previous_iy = np.array(self.pairings)[::-1][index-1]
            cost_matrix[previous_ix, previous_iy] = self.cost_matrix[previous_ix, previous_iy] # repair the cost matrix

            ix,iy = np.array(self.pairings)[::-1][index]
            cost_matrix[ix, iy] = 1e10 # put prohibitive cost
            
            # recompute DTW with this modified cost matrix
            DTW = np.ones((n+1,m+1))*np.inf
            DTW[0,0] = 0
            for i in range(n):
                for j in range(m):
                    cost = cost_matrix[i,j]
                    additionnal_cost = min(DTW[i+1,j], DTW[i,j+1], DTW[i, j])
                    DTW[i+1,j+1] = cost + additionnal_cost
            score = DTW[n,m]
            if debug: DTWs[index,:,:] = DTW

            minimal_additionnal_cost = score - self.score
            global_constraints[index] = minimal_additionnal_cost

        return global_constraints if not debug else (global_constraints, DTWs)
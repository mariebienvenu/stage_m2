
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import plotly.graph_objects as go

from app.curve import Curve
import app.maths_utils as m_utils
import app.visualisation as vis
from app.color import Color


def spot(half_size=4, max=100, min=5, profile='linear'):
    if profile != "linear" : raise NotImplementedError(f"Profiles others than linear have not been implmemented. Asked for {profile}.")
    spot = np.zeros((2*half_size+1, 2*half_size+1))
    for i in range(2*half_size+1):
        for j in range(2*half_size+1):
            dist = np.sqrt((i-half_size)**2 + (j-half_size)**2)
            if dist <= half_size:
                spot[i,j] = max - dist/half_size*(max-min)
    return spot



class DynamicTimeWarping:

    def __init__(self, curve1:Curve, curve2:Curve, normalize=True, use_spot=False):
        self.curve1 = deepcopy(curve1)
        self.curve2 = deepcopy(curve2)
        if normalize: self.curve1.normalize(), self.curve2.normalize()
        self.times1, self.times2 = curve1.get_times(), curve2.get_times()
        self.compute()

        self.bijection = (
            np.array([self.times1[i] for i,j in self.pairings]),
            np.array([self.times2[j] for i,j in self.pairings])
        )

        self.local_processed = False
        self.global_processed = False
        self.global_spot_processed = False
        self.filtered_indexes_done = False
        self.use_spot = use_spot
    

    def compute(self, eps=1e-11) : #1e-11 if distances is float64, Ae-5 if float32
        self.values1, self.values2 = np.ravel(self.curve1.get_values()), np.ravel(self.curve2.get_values())
        n,m = self.values1.size, self.values2.size
        self.cost_matrix = np.array([[abs(self.values1[i] - self.values2[j]) for j in range(m)] for i in range(n)]) # coût d'une paire = distance entre valeurs
        self.DTW = DynamicTimeWarping.distances(self.cost_matrix)
        self.score = self.DTW[n,m]
        self.pairings = DynamicTimeWarping.shortest_path(self.DTW, self.cost_matrix)
        assert abs(sum([self.cost_matrix[i,j] for i,j in self.pairings]) - self.score)<eps, f"DTW score does not match :\n\t final distance: {self.score} \n\t path cumulative cost: {sum([self.cost_matrix[i,j] for i,j in self.pairings])}"


    @property
    def cost_along_shortest_path(self):
        return np.array([self.cost_matrix[i,j] for i,j in self.pairings])
    

    @staticmethod
    def distances(costs:np.ndarray, distances=None, start_i=0, start_j=0):
        n,m = costs.shape
        if distances is None:
            distances = np.ones((n+1,m+1), dtype=np.float64)*np.inf
            distances[0,0] = 0
        for i in range(start_i, n):
            for j in range(start_j, m):
                cost = costs[i,j]
                additionnal_cost = min(distances[i+1,j], distances[i,j+1], distances[i, j])
                distances[i+1,j+1] = cost + additionnal_cost
        return distances


    @staticmethod
    def shortest_path(distances:np.ndarray, costs:np.ndarray):
        n,m = costs.shape
        eps = 1e-12 #1e-12 if distances in float64, 1e-5 if float32
        reverse_path = [[n-1,m-1]]
        i,j = n,m
        while i>1 or j>1:
            current = distances[i,j]
            precedent_cost = current - costs[i-1, j-1]
            if abs(distances[i-1, j-1] - precedent_cost) < eps:
                i -= 1
                j -= 1
            elif abs(distances[i, j-1] - precedent_cost) < eps:
                j -= 1
            elif abs(distances[i-1, j] - precedent_cost) < eps:
                i -= 1
            else:
                print(f"i:{i}, j:{j}, threhold:{eps}")
                print(f" Candidate [i-1, j-1]: {abs(distances[i-1, j-1] - precedent_cost)}")
                print(f" Candidate [  i, j-1]: {abs(distances[i, j-1] - precedent_cost)}")
                print(f" Candidate [i-1,   j]: {abs(distances[i-1, j] - precedent_cost)}")
                raise TimeoutError("Stuck in infinite 'while' loop in DynamicTimeWarping.shortest_path()...")
            reverse_path.append([i-1,j-1])
        path = reverse_path[::-1]
        return path


    def local_constraints(self, window_size=10):
        if self.local_processed: return self._local_constraints
        range_x = len(self.curve1)
        range_y = len(self.curve2)
        N = self.bijection[0].size
        self._local_constraints = np.zeros((N))
        for i in range(1, N-1):
            ix,iy = self.pairings[i]
            center_cost = self.cost_matrix[ix, iy] # best cost (globally)
            w_size = min(window_size, ix, iy, range_x-ix, range_y-iy)
            upper_costs = self.cost_matrix[ix+1:ix+w_size, iy] + self.cost_matrix[ix, iy+1:iy+w_size]
            lower_costs = self.cost_matrix[ix-w_size:ix, iy] + self.cost_matrix[ix, iy-w_size:iy]
            alternative_costs = np.concatenate((upper_costs, lower_costs))
            minimal_additionnal_cost = np.min(alternative_costs) - center_cost if alternative_costs.size>0 else 0
            self._local_constraints[i] = max(0,minimal_additionnal_cost)
        self.local_processed = True
        return self._local_constraints
    

    def global_constraints(self):
        res = self.global_constraints_spot(spot_half_size=int(self.use_spot)) if self.use_spot else self.global_constraints_no_spot()
        self._global_constraints = res
        ## TODO revoir ça, c'est super mal fait
        if self.use_spot:
            self.global_constraints_alternative_paths = self.global_constraints_spot_alternative_paths
            self.global_constraints_distances = self.global_constraints_spot_distances
        else:
            self.global_constraints_alternative_paths = self.global_constraints_no_spot_alternative_paths
            self.global_constraints_distances = self.global_constraints_no_spot_distances
        return res
    

    def global_constraints_no_spot(self):
        """Computes the cost each pair contributed to save to the final score, using DTW"""
        if self.global_processed: return self._global_constraints_no_spot
        N = self.bijection[0].size
        self.global_constraints_no_spot_distances = np.zeros((N, self.values1.size+1, self.values2.size+1))
        self.global_constraints_no_spot_alternative_paths = [None for _ in range(N)]

        self._global_constraints_no_spot = np.zeros((N))
        cost_matrix = np.copy(self.cost_matrix)
        distances = DynamicTimeWarping.distances(cost_matrix)

        for index in tqdm(range(N-2, 0, -1), desc="Global constraint on DTW computation"):

            previous_ix, previous_iy = self.pairings[index+1]
            cost_matrix[previous_ix, previous_iy] = self.cost_matrix[previous_ix, previous_iy] # repair the cost matrix

            ix,iy = self.pairings[index]
            cost_matrix[ix, iy] = 1e10 # put prohibitive cost
            
            distances = DynamicTimeWarping.distances(cost_matrix, distances, start_i=ix, start_j=iy) # recompute DTW with this modified cost matrix using previous data as much as possible
            
            self.global_constraints_no_spot_distances[index,:,:] = distances
            self.global_constraints_no_spot_alternative_paths[index] = DynamicTimeWarping.shortest_path(distances, cost_matrix)

            alternative_score = distances[-1,-1]
            minimal_additionnal_cost = alternative_score - self.score
            self._global_constraints_no_spot[index] = minimal_additionnal_cost

        self.global_processed = True
        return self._global_constraints_no_spot
    

    def global_constraints_spot(self, spot_half_size=4, spot_max=100, spot_min=5):
        """Computes the cost each pair contributed to save to the final score, using DTW"""
        if self.global_spot_processed: return self._global_constraints_spot
        N = self.bijection[0].size
        self.global_constraints_spot_distances = np.zeros((N, self.values1.size+1, self.values2.size+1))
        self.global_constraints_spot_alternative_paths = [None for _ in range(N)]
        spot_mtx = spot(spot_half_size, spot_max, spot_min)

        self._global_constraints_spot = np.zeros((N))
        cost_matrix = np.copy(self.cost_matrix)
        distances = DynamicTimeWarping.distances(cost_matrix)

        n,m = cost_matrix.shape

        for index in tqdm(range(N-1-spot_half_size, spot_half_size, -1), desc="Global constraint on DTW computation"):

            previous_ix, previous_iy = self.pairings[index+1]
            x1, y1, x2, y2 = max(previous_ix-spot_half_size, 0), max(previous_iy-spot_half_size, 0), min(previous_ix+spot_half_size+1, n), min(previous_iy+spot_half_size+1, m)
            cost_matrix[x1:x2, y1:y2] = self.cost_matrix[x1:x2, y1:y2] # repair the cost matrix

            ix, iy = self.pairings[index]
            x1, y1, x2, y2 = max(ix-spot_half_size, 0), max(iy-spot_half_size, 0), min(ix+spot_half_size+1, n-1), min(iy+spot_half_size+1, m-1)
            w, h = min(x2-ix,ix-x1), min(y2-iy, iy-y1)
            real_half_size = min(w,h)
            if real_half_size != spot_half_size :
                cost_matrix[ix-real_half_size:ix+real_half_size+1, iy-real_half_size:iy+real_half_size+1] += spot(half_size=real_half_size)
            else:
                cost_matrix[ix-spot_half_size:ix+spot_half_size+1, iy-spot_half_size:iy+spot_half_size+1] += spot_mtx # put prohibitive cost
            
            distances = DynamicTimeWarping.distances(cost_matrix, distances, start_i=ix-spot_half_size, start_j=iy-spot_half_size) # recompute DTW with this modified cost matrix using previous data as much as possible
            
            self.global_constraints_spot_distances[index,:,:] = distances
            self.global_constraints_spot_alternative_paths[index] = DynamicTimeWarping.shortest_path(distances, cost_matrix)

            alternative_score = distances[-1,-1]
            minimal_additionnal_cost = alternative_score - self.score
            self._global_constraints_spot[index] = minimal_additionnal_cost

        self.global_spot_processed = True
        return self._global_constraints_spot
    

    def alternate_path_differences(self):
        global_constraints = self.global_constraints()
        alternative_paths = self.global_constraints_alternative_paths
        self._alternate_path_differences = np.zeros_like(global_constraints)
        timespan1, timespan2 = self.curve1.time_range[1] - self.curve1.time_range[0], self.curve2.time_range[1] - self.curve2.time_range[0]
        for i, path in enumerate(alternative_paths):
            if path is None:
                continue
            bij_x, bij_y = np.array([self.times1[i] for i,_ in path]), np.array([self.times2[j] for _,j in path])
            self._alternate_path_differences[i] = m_utils.distL1(bij_x,bij_y, self.bijection[0], self.bijection[1])
        self._alternate_path_differences *= 100/timespan1/timespan2
        return self._alternate_path_differences
    

    def filtered_indexes(self, use_global=True, use_constraint_local_maximum=True, constraint_threshold=2, area_difference_threshold=0.1, time_window=15): # 1/4sec #time_window was 5 before...
        ## TODO dtw.filtered_indexes() -- not tested yet
        if self.filtered_indexes_done : return self._filtered_indexes
        pair_indexes = list(range(1, len(self.pairings)-1))
        constraints = self.global_constraints() if use_global else self.local_constraints()
        alternate_path_differences = self.alternate_path_differences()
        self.is_index_constrained_enough = [constraints[index]>constraint_threshold for index in pair_indexes]
        self.is_index_similar_enough = [alternate_path_differences[index-1]<area_difference_threshold for index in pair_indexes]
        self.is_index_local_max = [constraints[index]>=max(constraints[index-1], constraints[index+1]) if use_constraint_local_maximum else True for index in pair_indexes]
        self._filtered_indexes = [0] + [
            index for i,index in enumerate(pair_indexes) 
            if self.is_index_constrained_enough[i] 
            and self.is_index_similar_enough[i] 
            and self.is_index_local_max[i]
        ] + [len(constraints)-1]
        short_constraints = [e for i,e in enumerate(constraints) if i in self._filtered_indexes]
        keep = [
            all([
                abs(self.bijection[0][i]-self.bijection[0][j])>time_window # before, it was just abs(i-j)>time_window, but it actually makes no sense in term of units ?
                or abs(self.bijection[1][i]-self.bijection[1][j])>time_window 
                or short_constraints[jdx]<short_constraints[idx] 
                or (short_constraints[jdx]==short_constraints[idx] and j>i) 
                or i==j 
                for jdx,j in enumerate(self._filtered_indexes)
            ]) for idx,i in enumerate(self._filtered_indexes)
        ]
        debug=0
        self._filtered_indexes = [i for i,k in zip(self._filtered_indexes,keep) if k]
        self.filtered_indexes_done = True
        return self._filtered_indexes
    

    def detect_limitation(self, cost_threshold=7, constraint_threshold=2, area_difference_threshold=0.1, local_window_size=20, refine=True):
        """Local window_size is the size of the voisinage taken into account, in index space. It is not a time or even necessarily a number of frames."""
        cost_along_path = self.cost_along_shortest_path
        kept_indexes = self.filtered_indexes(constraint_threshold=constraint_threshold, area_difference_threshold=area_difference_threshold)
        problematic_indexes = []
        for index, cost in enumerate(cost_along_path):
            if cost>cost_threshold:
                if not any([kept>=index-local_window_size and kept<=index+local_window_size for kept in kept_indexes]):
                    problematic_indexes.append(index)
        if not refine: return problematic_indexes
        keep = [False for _ in problematic_indexes]
        for i, problematic_index in enumerate(problematic_indexes):
            neighbouring_indexes = [index for index in problematic_indexes if abs(index-problematic_index)<=local_window_size]
            if len(neighbouring_indexes)==0:
                keep[i] = True
            else:
                if cost_along_path[problematic_index] >= max(cost_along_path[neighbouring_indexes]):
                    keep[i] = True
        problematic_indexes = [index for i, index in enumerate(problematic_indexes) if keep[i]]
        return problematic_indexes


    def make_map(self, add_path=True, path_color=None, fig=None) -> go.Figure:
        if fig is None: fig=go.Figure()
        vis.add_heatmap(self.cost_matrix, fig=fig)
        pairs = self.pairings
        if add_path:
            vis.add_curve(y=np.array(pairs)[:,0]-pairs[0][0], x=np.array(pairs)[:,1]-pairs[0][1], color=path_color, fig=fig)
        return fig
    

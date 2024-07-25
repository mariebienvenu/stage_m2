
from enum import Enum
from copy import deepcopy

import numpy as np
import scipy.sparse.linalg as linalg

from app.animation import Animation
from app.curve import Attributes_Name, Curve
import app.warping as W
import app.visualisation as vis


class TangentSemantic(Enum):
    NOTHING = 0
    KEEP_ANGLE = 1
    KEEP_LENGTH = 2
    KEEP_BOTH = 3


class SemanticRetiming:


    #REGULARIZATION_WEIGHT = 1 # Global factor, will decide how much the regularization will weigh in compared to the matches
    ALIGNMENT_WEIGHT = 1 # Should be <=1 # This is not a good name : alignment are preserved natively. This is meant to  help enforce symmetry inside a key (between the left and right tangent length).
    BROKEN_WEIGHT = 0.2 # Should be <=1
    NEIGHBOURS_WEIGHT = 0.2 #0.7 # Should be <=1 ## Actually, by construction, there is no need to enforce symmetries between keys as they will be kept natively
    #But it will help still : it will penalize neighbour which are changing from base in a different manner, but not penalize neigbours which are changing from base in the same manner.


    def reset_weights():
        SemanticRetiming.ALIGNMENT_WEIGHT = 0.5
        SemanticRetiming.BROKEN_WEIGHT = 0.3
        SemanticRetiming.NEIGHBOURS_WEIGHT = 1


    def __init__(self, animation:Animation, channels:list[str], matches:np.ndarray, tangent_semantic=TangentSemantic.KEEP_ANGLE):
        """Matches should be ordered in time"""
        if tangent_semantic.value != TangentSemantic.KEEP_ANGLE.value: raise NotImplementedError(f"Tangent semantic not implemented yet. Expected TangentSemantic.KEEP_ANGLE, got {tangent_semantic}")
        self.matches = matches # array of size (n,2) with the match "times"
        self.animation = animation
        self.channels = channels
        self.match_count = matches.shape[0]
        self.tangent_semantic = tangent_semantic # not used currently
        self.is_processed = False


    def process(self, force=False, interpolation="linear", regularization_weight=1):
        if self.is_processed and not force: return self.new_animation
        self.snap()
        self.make_retiming_warp(interpolation=interpolation)
        regul, broken, aligned, neighbour = regularization_weight, SemanticRetiming.BROKEN_WEIGHT, SemanticRetiming.ALIGNMENT_WEIGHT, SemanticRetiming.NEIGHBOURS_WEIGHT
        self.optimize_tangent_scaling(broken_weight=regul*broken, aligned_weight=regul*aligned, neighbours_weight=regul*neighbour)
        self.make_new_animation()
        self.is_processed = True
        return self.new_animation

    
    def snap(self, to_integer=True, attraction=50):
        all_times = np.concatenate([curve.get_times() for curve in self.animation if curve.fullname in self.channels]).astype(int)
        times = np.unique(all_times)
        histogram = np.bincount(all_times)
        occurences = np.array([histogram[t] for t in times]) # if channels is of length 1, this should be all 1
        self.snapped_matches = np.copy(self.matches)
        for i in range(self.match_count):
            time_ref = self.matches[i,0]
            distance = np.abs(times-time_ref)
            metric = distance - attraction*(occurences/len(self.channels)) # TODO this is a homemade metric, probably very much not what we want...
            index = np.argmin(metric)
            time_snapped = times[index]
            self.snapped_matches[i,0] = time_snapped
            #self.snapped_matches[i,1] += time_snapped-time_ref # après discussion avec Damien : non
        if np.unique(self.snapped_matches[:,0]).size != self.match_count:
            debug = 0
            raise AssertionError("Two matches converged to the same keyframe, this is not handled. Will crash.")
        if to_integer: self.snapped_matches = self.snapped_matches.astype(int)

    
    @property
    def snapped_times_reference(self):
        return self.snapped_matches[:,0]
    
    @property
    def snapped_times_target(self):
        return self.snapped_matches[:,1]

    def make_retiming_warp(self, interpolation='linear'):
        self.retiming_warp = W.make_warp(dimension=1, interpolation=interpolation, X_in=self.snapped_times_reference, X_out=self.snapped_times_target)
    

    def optimize_tangent_scaling(self, broken_weight=1, aligned_weight=1, neighbours_weight=1):
        ## We are going to get the left and right scaling that should be applied to the keys that are on snapped times
        ## Taking into account some form of regularization in time : 1D chain with unknown only interacting with neigbours
        left_zipper = zip(self.snapped_times_target[1:], self.snapped_times_target[:-1], self.snapped_times_reference[1:], self.snapped_times_reference[:-1])
        self.basic_left = [1]+[(y-y_bef)/(x-x_bef) for y,y_bef,x,x_bef in left_zipper]
        right_zipper = zip(self.snapped_times_target[:-1], self.snapped_times_target[1:], self.snapped_times_reference[:-1], self.snapped_times_reference[1:])
        self.basic_right = [(y_aft-y)/(x_aft-x) for y,y_aft,x,x_aft in right_zipper]+[1]

        n_unknown = 2*self.match_count
        initial_conditions = np.array([self.basic_left[i//2] if i%2==0 else self.basic_right[i//2] for i in range(n_unknown)])
        alignments = np.zeros((self.match_count))
        for curve in self.animation:
            if curve.fullname in self.channels:
                are_aligned = curve.are_tangents_aligned()
                times = curve.get_times()
                for (time, is_aligned) in zip(times, are_aligned):
                    if time in self.snapped_times_reference and is_aligned:
                        match_index = list(self.snapped_times_reference).index(time)
                        alignments[match_index] += 1
        #np.sum(np.array([curve.are_tangents_aligned() for curve in self.animation if curve.fullname in self.channels]), axis=0)
        regularization_weights = [(aligned_weight if alignments[i//2]>0 else broken_weight) if i%2==0 else neighbours_weight for i in range(n_unknown-1)]

        # We are going to find X such as to minimize ||AX-B||²
        A = np.zeros((2*n_unknown-1, n_unknown))
        for i in range(n_unknown):
            A[i,i] = 1
            if i != n_unknown-1:
                A[n_unknown+i, i] = regularization_weights[i]
                A[n_unknown+i, i+1] = -regularization_weights[i]
        B = np.zeros((2*n_unknown-1))
        for i in range(n_unknown):
            B[i] = initial_conditions[i]

        ## Doing the solving
        least_squares = linalg.lsqr(A=A, b=B)
        X = least_squares[0] # there are other outputs... Like 9 more, mostly some info about how the algo went

        self.new_left = [X[2*i] for i in range(self.match_count)]
        self.new_right = [X[2*i+1] for i in range(self.match_count)]

    
    def left_tangent_operator(self, key_time_reference, tangent_vector):
        ## This implements the "keep_angle" semantic
        if key_time_reference > self.snapped_times_reference[-1]:
            return self.new_right[-1]*tangent_vector
        if key_time_reference <=     self.snapped_times_reference[0]:
            return self.new_left[0]*tangent_vector
        current_index = 0
        while current_index<self.match_count and key_time_reference > self.snapped_times_reference[current_index]:
            current_index += 1
        if key_time_reference == self.snapped_times_reference[current_index]:
            return self.new_left[current_index]*tangent_vector
        scale_before = self.new_left[current_index]
        scale_after = self.new_right[current_index-1]
        time_before, time_after = self.snapped_times_reference[current_index-1], self.snapped_times_reference[current_index]
        correct_scale = scale_before + (scale_after-scale_before)*(key_time_reference-time_before)/(time_after-time_before)
        return correct_scale*tangent_vector
    

    def right_tangent_operator(self, key_time_reference, tangent_vector):
        ## This implements the "keep_angle" semantic
        if key_time_reference >= self.snapped_times_reference[-1]:
            return self.new_right[-1]*tangent_vector
        if key_time_reference < self.snapped_times_reference[0]:
            return self.new_left[0]*tangent_vector
        current_index = self.match_count-1
        while current_index>0 and key_time_reference < self.snapped_times_reference[current_index]:
            current_index -= 1
        if key_time_reference == self.snapped_times_reference[current_index]:
            return self.new_right[current_index]*tangent_vector
        scale_before = self.new_right[current_index]
        scale_after = self.new_left[current_index+1]
        time_before, time_after = self.snapped_times_reference[current_index], self.snapped_times_reference[current_index+1]
        correct_scale = scale_before + (scale_after-scale_before)*(key_time_reference-time_before)/(time_after-time_before)
        return correct_scale*tangent_vector
    

    def make_new_animation(self):
        self.new_animation = Animation()
        for curve in self.animation:
            if curve.fullname not in self.channels:
                self.new_animation.append(curve)
            else:
                new_curve = deepcopy(curve)
                times = curve.get_times()
                values = curve.get_values()
                co = np.vstack((times, values)).T
                left_tangent_pos = np.vstack((curve.get_attribute('handle_left_x'), curve.get_attribute('handle_left_y'))).T
                right_tangent_pos = np.vstack((curve.get_attribute('handle_right_x'), curve.get_attribute('handle_right_y'))).T
                left_tangent_vector = left_tangent_pos - co
                right_tangent_vector = right_tangent_pos - co

                new_left_tangent_vector = np.array([self.left_tangent_operator(time, tangent) for time, tangent in zip(times, left_tangent_vector)])
                new_right_tangent_vector = np.array([self.right_tangent_operator(time, tangent) for time, tangent in zip(times, right_tangent_vector)])

                new_curve.apply_spatio_temporal_warp(self.retiming_warp)
                new_times, new_values = new_curve.get_times(), new_curve.get_values()
                new_curve.set_attribute("handle_left_x", new_left_tangent_vector[:,0]+new_times)
                new_curve.set_attribute("handle_left_y", new_left_tangent_vector[:,1]+new_values)
                new_curve.set_attribute("handle_right_x", new_right_tangent_vector[:,0]+new_times)
                new_curve.set_attribute("handle_right_y", new_right_tangent_vector[:,1]+new_values)

                self.new_animation.append(new_curve)
        return self.new_animation
    

    def diagram(self):

        fig = vis.add_curve(y=self.basic_left, x=self.snapped_times_reference, name="Left basic warp")
        vis.add_curve(y=self.basic_right, x=self.snapped_times_reference, name="Right basic warp", fig=fig)

        vis.add_curve(y=self.new_left, x=self.snapped_times_reference, name="Left new warp", fig=fig)
        vis.add_curve(y=self.new_right, x=self.snapped_times_reference, name="Right new warp", fig=fig)

        return fig
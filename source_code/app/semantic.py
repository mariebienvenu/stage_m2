
from enum import Enum
from copy import deepcopy

import numpy as np
import scipy.sparse.linalg as linalg

from app.animation import Animation
from app.curve import Attributes_Name, Curve
import app.warping as W


class TangentSemantic(Enum):
    NOTHING = 0
    KEEP_ANGLE = 1
    KEEP_LENGTH = 2
    KEEP_BOTH = 3


class SemanticRetiming: ## TODO wirte dedicated test file


    def __init__(self, animation:Animation, channels:list[str], matches:np.ndarray, tangent_semantic=TangentSemantic.KEEP_ANGLE):
        """Matches should be ordered in time"""
        if tangent_semantic.value != TangentSemantic.KEEP_ANGLE.value: raise NotImplementedError(f"Tangent semantic not implemented yet. Expected TangentSemantic.KEEP_ANGLE, got {tangent_semantic}")
        self.matches = matches # array of size (n,2) with the match "times"
        self.animation = animation
        self.channels = channels
        self.match_count = matches.shape[0]
        self.tangent_semantic = tangent_semantic # not used currently
        self.is_processed = False


    def process(self, force=False):
        if self.is_processed and not force: return self.new_animation
        self.snap()
        self.make_retiming_warp()
        self.optimize_tangent_scaling()
        self.make_new_animation()
        self.is_processed = True
        return self.new_animation

    
    def snap(self, to_integer=True, attraction=0):
        all_times = np.concatenate([curve.get_times() for curve in self.animation if curve.fullname in self.channels])
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
    

    def optimize_tangent_scaling(self, regularization_weight=1):
        ## We are going to get the left and right scaling that should be applied to the keys that are on snapped times
        ## Taking into account some form of regularization in time : 1D chain with unknown only interacting with neigbours
        left_zipper = zip(self.snapped_times_target[1:], self.snapped_times_target[:-1], self.snapped_times_reference[1:], self.snapped_times_reference[:-1])
        basic_left = [1]+[(y-y_bef)/(x-x_bef) for y,y_bef,x,x_bef in left_zipper]
        right_zipper = zip(self.snapped_times_target[:-1], self.snapped_times_target[1:], self.snapped_times_reference[:-1], self.snapped_times_reference[1:])
        basic_right = [(y_aft-y)/(x_aft-x) for y,y_aft,x,x_aft in right_zipper]+[1]

        n_unknown = 2*self.match_count
        initial_conditions = np.array([basic_left[i//2] if i%2==0 else basic_right[i//2] for i in range(n_unknown)])

        # We are going to find X such as to minimize ||AX-B||²
        A = np.zeros((2*n_unknown-1, n_unknown))
        for i in range(n_unknown):
            A[i,i] = 1
            A[n_unknown+i, i] = regularization_weight
            if i!=0:
                A[n_unknown+i, i-1] = -regularization_weight
        B = np.zeros((2*n_unknown-1))
        for i in range(n_unknown):
            B[i] = initial_conditions[i]

        ## Doing the solving
        X = linalg.lsqr(A=A, b=B)

        self.new_left = [X[2*i] for i in range(self.match_count)]
        self.new_right = [X[2*i+1] for i in range(self.match_count)]

    
    def left_tangent_operator(self, key_time_reference, tangent_vector):
        ## This implements the "keep_angle" semantic
        if key_time_reference >= self.snapped_times_reference[-1]:
            return self.new_right[-1]*tangent_vector
        if key_time_reference < self.snapped_times_reference[0]:
            return self.new_left[0]*tangent_vector
        current_index = 0
        while current_index<self.match_count and key_time_reference > self.snapped_times_reference[current_index]:
            current_index += 1
        if key_time_reference == self.snapped_times_reference[current_index]:
            return self.new_left[current_index]*tangent_vector
        scale_before = self.new_left[current_index-1]
        scale_after = self.new_right[current_index]
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
        scale_before = self.new_left[current_index]
        scale_after = self.new_right[current_index+1]
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
                left_tangent_pos = np.vstack((curve.get_attribute('tangent_left_x'), curve.get_attribute('tangent_left_y'))).T
                right_tangent_pos = np.vstack((curve.get_attribute('tangent_right_x'), curve.get_attribute('tangent_right_y'))).T
                left_tangent_vector = left_tangent_pos - co
                right_tangent_vector = right_tangent_pos - co

                new_left_tangent_vector = np.array([self.left_tangent_operator(time, tangent) for time, tangent in zip(times, left_tangent_vector)])
                new_right_tangent_vector = np.array([self.right_tangent_operator(time, tangent) for time, tangent in zip(times, right_tangent_vector)])

                new_curve.apply_spatio_temporal_warp(self.retiming_warp)
                new_times, new_values = new_curve.get_times(), new_curve.get_values()
                new_curve.set_attribute("tangent_left_x", new_left_tangent_vector[:,0]+new_times)
                new_curve.set_attribute("tangent_left_y", new_left_tangent_vector[:,1]+new_values)
                new_curve.set_attribute("tangent_right_x", new_right_tangent_vector[:,0]+new_times)
                new_curve.set_attribute("tangent_right_y", new_right_tangent_vector[:,1]+new_values)

                self.new_animation.append(new_curve)
        return self.new_animation

    """def tangent_time_stretch(self, epsilon=1e-10):
        keep_aligned = [False for _ in range(self.match_count)]
        symmetry_to_left = [False for _ in range(self.match_count)]
        symmetry_to_right = [False for _ in range(self.match_count)]
        for channel in self.channels:
            curve = self.animation.find(channel)
            are_aligned = curve.are_tangents_aligned()
            left_tangent_vector = np.hstack((curve.get_attribute('left_tangent_x')-curve.get_attribute('time'), curve.get_attribute('left_tangent_y')-curve.get_attribute('value')))
            right_tangent_vector = np.hstack((curve.get_attribute('right_tangent_x')-curve.get_attribute('time'), curve.get_attribute('right_tangent_y')-curve.get_attribute('value')))
            lshape = left_tangent_vector.shape
            debug=1
            for i,time in enumerate(self.snapped_times_reference):                
                times = curve.get_times()
                index = list(times).index(time)
                if are_aligned[index]:
                    keep_aligned[i] = True
                if i>=1:
                    previous_index = list(times).index(self.snapped_times_reference[i-1])
                    lx,ly = right_tangent_vector[previous_index,:]
                    rx,ry = left_tangent_vector[index,:]
                    if abs(ly-ry)<epsilon and abs(lx+rx)<epsilon:
                        symmetry_to_left[index]=True
                        symmetry_to_right[previous_index] = True
        debug=2
        left_zipper = zip(self.snapped_times_target[1:], self.snapped_times_target[:-1], self.snapped_times_reference[1:], self.snapped_times_reference[:-1])
        basic_left_stretch = [1]+[(y-y_bef)/(x-x_bef) for y,y_bef,x,x_bef in left_zipper]
        right_zipper = zip(self.snapped_times_target[:-1], self.snapped_times_target[1:], self.snapped_times_reference[:-1], self.snapped_times_reference[1:])
        basic_right_stretch = [(y_aft-y)/(x_aft-x) for y,y_aft,x,x_aft in right_zipper]+[1]

        ## We will use scipy.optimize.linprog to solve our problem, which is to enforce alignment and symmetry while minimizing change.

        self.left_time_stretch = []
        self.right_time_stretch = []
        for i,channel in enumerate(self.channels):
            self.left_time_stretch.append([])
            self.right_time_stretch.append([])
            snapped_index = 0
            curve = self.animation.find(channel)
            times = curve.get_times()
            for j in range(len(curve)):
                time = times[j]
                if time in self.snapped_times:
                    snapped_index = list(self.snapped_times).index(time)
                    basic_left_stretch = 
                    basic_right_stretch =
                    if keep_aligned[snapped_index]:
                        if symmetry_to_left[snapped_index]:

                        else:
                        if symmetry_to_right[snapped_index]:
                        else:
                    else:
                        if symmetry_to_left[snapped_index]:
                        else:
                        if symmetry_to_right[snapped_index]:
                        else:
                else:

                self.left_time_stretch[i].append(left_stretch)
                self.right_time_stretch[i].append(right_stretch)
        return
    

    def tangent_value_stretch(self, mode=TangentSemantic.KEEP_ANGLE):
        return # TODO
    
    def make_new_animation(self):
        self.new_animation = None
        # TODO"""
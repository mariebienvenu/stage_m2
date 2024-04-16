
import cv2
import numpy as np
from enum import Enum


class Measure(Enum):
    MAGNITUDE_MEAN = 0
    MAGNITUDE_STD = 1
    ANGLE_MEAN = 2
    ANGLE_STD = 3


class OpticalFlow(np.ndarray):

    def __new__(cls, array:np.ndarray, use_degrees=False):
        obj = np.asarray(array).view(cls)
        obj.use_degrees = use_degrees
        obj.polar = OpticalFlow.cartesian_to_polar(obj.x, obj.y, degrees=use_degrees)
        return obj
    

    def __array_finalize__(self, obj):
        if obj is None : return
        self.use_degrees = getattr(obj, 'use_degrees', None)
        self.polar = getattr(obj, 'polar', None)


    @staticmethod
    def compute_oflow(im1, im2, winsize=15, levels=3, iterations=3, poly_n=5, poly_sigma=1.3, use_degrees=False):
        assert OpticalFlow.is_grayscale(im1) and OpticalFlow.is_grayscale(im2), "Provided images are not in grayscale."
        oflow = cv2.calcOpticalFlowFarneback(
            im1,
            im2,
            flow=None,
            pyr_scale=0.5,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0
            )
        return OpticalFlow(oflow, use_degrees=use_degrees)
    

    @property
    def x(self):
        return self[...,0]
    
    @property
    def y(self):
        return self[...,1]

    @property
    def magnitude(self):
        return self.polar[0]
    
    @property
    def angle(self):
        res = self.polar[1]
        max_angle = 180 if self.use_degrees else np.pi
        res[res>max_angle] -= 2*max_angle
        return res


    def _get_threshold(self, proportion):
        flattened = np.ravel(self)
        kth_index = int(np.size(flattened)*proportion)
        ordered = np.partition(flattened, kth_index)
        threshold = ordered[kth_index]
        return threshold


    def get_mask(self, background_proportion=0.0):
        threshold = OpticalFlow._get_threshold(self.magnitude, background_proportion)
        mask = np.where(self.magnitude>threshold, 1, 0)
        return mask


    def make_oflow_image(self): # Used for visualisation purposes ; encodes polar into hsv
        hsv = np.zeros((self.magnitude.shape[0], self.magnitude.shape[1], 3), dtype=np.uint8)
        hsv[:,:, 0] = self.angle*180/np.pi/2 if not self.use_degrees else self.angle/2 # Hue between 0 and ?
        hsv[:,:, 1] = 255 # Saturation between 0 and 255
        hsv[:,:, 2] = cv2.normalize(self.magnitude, None, 0, 255, cv2.NORM_MINMAX) # Value between 0 and 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return bgr
    

    def get_measure(self, measure:Measure, mask=None):
        mask = np.ones_like(self.magnitude) if mask is None else mask
        if measure == Measure.MAGNITUDE_MEAN:
            return np.mean(self.magnitude[mask>0])
        elif measure == Measure.MAGNITUDE_STD:
            return np.std(self.magnitude[mask>0])
        elif measure == Measure.ANGLE_MEAN:
            return np.mean(self.magnitude[mask>0]*self.angle[mask>0])/np.mean(self.magnitude[mask>0])
        elif measure == Measure.ANGLE_STD:
            return np.std(self.angle[mask>0])
     
        
    @staticmethod
    def is_grayscale(image:np.ndarray):
        return len(image.shape)==2 or (len(image.shape)==3 and image.shape[2]==1)
    
    @staticmethod
    def cartesian_to_polar(x, y, degrees=False):
        return cv2.cartToPolar(x, y, angleInDegrees=degrees)

    @staticmethod
    def polar_to_cartesian(magnitude, angle, degrees=False):
        return cv2.polarToCart(magnitude, angle, angleInDegrees=degrees)
    


def get_crop(frame_times, magnitude_mean, threshold=0.1, padding_out=10, padding_in=3, patience=0): # oflow.get_crop() -- TODO move to Curve.py 
    ## TODO oflow.get_crop() --  magnitude_means should actually be normalized (right now its scale depends heavily on background proportion) so that this threshold can stay the same.
    ## TODO oflow.get_crop() -- passer à un input de type curve ? et ajouter un constant=0 pour gérer des courbes qui plateaux à autre chose que 0
    start = padding_out
    stop = magnitude_mean.size-padding_out
    used_patience_left = 0
    used_patience_right = 0
    for i, (mag, opp_mag) in enumerate(zip(magnitude_mean, magnitude_mean[::-1])):
        if start == i:
            if not abs(mag)<threshold and used_patience_left<patience:
                start += 1
                used_patience_left += 1
            elif abs(mag)<threshold:
                start += 1
                used_patience_left = 0
        if stop == magnitude_mean.size-i:
            if not abs(opp_mag)<threshold and used_patience_right<patience:
                stop -= 1
                used_patience_right += 1
            elif abs(opp_mag)<threshold:
                stop -= 1
                used_patience_right = 0
    if start==frame_times.size or stop==0:
        return (frame_times[0], frame_times[0]) # found nothing good to keep...
    start -= padding_in + used_patience_left
    stop += padding_in + used_patience_right
    start = max(0, start)
    stop = min(stop, magnitude_mean.size)
    assert start <= stop, "Problem encountered when autocropping."
    return (frame_times[start], frame_times[stop])

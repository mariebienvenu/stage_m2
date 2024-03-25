import cv2
import numpy as np

def compute_oflow(im1, im2, winsize=15, levels=3, iterations=3, poly_n=5, poly_sigma=1.3): #should be in grayscale
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
    return oflow

def cartesian_to_polar(array, degrees=False):
    return cv2.cartToPolar(array[..., 0], array[..., 1], angleInDegrees=degrees)

def _get_threshold(array, proportion):
    flattened = np.ravel(array)
    kth_index = int(np.size(flattened)*proportion)
    ordered = np.partition(flattened, kth_index)
    threshold = ordered[kth_index]
    return threshold

def get_mask_oflow(mag, background_proportion=0.97):
    threshold = _get_threshold(mag, background_proportion)
    mask = np.expand_dims(np.where(mag>threshold, 1, 0),axis=2)
    mask = np.concatenate((mask,mask),axis=2)
    return mask

def make_oflow_image(mag, ang): #magnitude, angle. Used for visualisation purposes
    hsv = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.uint8)
    hsv[:,:, 0] = ang*180/np.pi/2 # Hue between 0 and ?
    hsv[:,:, 1] = 255 # Saturation between 0 and 255
    hsv[:,:, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Value between 0 and 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def measure_oflow(magnitude, angle):
    return {
        "magnitude_mean":np.mean(magnitude),
        "magnitude_std":np.std(magnitude), 
        "angle_mean":np.mean(magnitude*angle)/np.mean(magnitude),
        "angle_std":np.std(magnitude*angle)/np.std(magnitude)
    }

def get_crop(magnitude_mean, threshold=2, padding_out=10, padding_in=3):
    start = padding_out
    stop = magnitude_mean.size-padding_out
    for i, (mag, opp_mag) in enumerate(zip(magnitude_mean, magnitude_mean[::-1])):
        if start == i and abs(mag)<threshold:
            start += 1
        if stop == magnitude_mean.size-i and abs(opp_mag)<threshold:
            stop-=1
    start -= padding_in
    stop += padding_in
    start = max(0, start)
    stop = min(stop, magnitude_mean.size)
    assert start <= stop, "Problem encountered when autocropping."
    return (start, stop)

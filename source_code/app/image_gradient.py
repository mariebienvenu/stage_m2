
import numpy as np
import cv2

def image_gradient(image:np.ndarray):
    grad_x = cv2.Sobel(image.astype(np.float64), ddepth=cv2.CV_64F, dx=1, dy=0)
    grad_y = cv2.Sobel(image.astype(np.float64), ddepth=cv2.CV_64F, dx=0, dy=1)
    return np.stack((grad_x, grad_y))

import cv2

class ImageProcessing:

    @staticmethod
    def none(image):
        return image
    
    @staticmethod
    def gray(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def red(image):
        return image[:,:,2]
    
    @staticmethod
    def green(image):
        return image[:,:,1]
    
    @staticmethod
    def blue(image):
        return image[:,:,0]
    
    @staticmethod
    def hue(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,0]
    
    @staticmethod
    def saturation(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,1]
    
    @staticmethod
    def value(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,2]
    
    @staticmethod
    def custom(image, r=0.33, g=0.33, b=0.33):
        return r*ImageProcessing.red(image) + g*ImageProcessing.green(image) + b*ImageProcessing.blue(image)
    
    @staticmethod
    def rgb(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
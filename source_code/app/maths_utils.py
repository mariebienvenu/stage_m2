
import numpy as np
import scipy.stats as stats
from scipy.integrate import RK45 as runge_kutta
import scipy.integrate as integrate

def derivee(array, step):
    ar = np.array(array)
    res = (ar[2:]-ar[:-2])/(2*step) # différences finies
    #left = res[:-1]
    #right = res[1:]
    #for i in range(left.size):
    #    if left[i]!=0 and right[i]!=0 and int(left[i]/abs(left[i])) != int(right[i]/abs(right[i])):
    #        res[i]=0 # sign change means an actual derivative of 0
    return res


def derivee_seconde(array, step): # TODO m_utils.derivee_seconde() -- à tester dans maths.py
    ar = np.array(array)
    res = (-2*ar[1:-1] + ar[:-2] + ar[2:])/step**2  # différences finies
    return res


def integrale1(array, step):
    arr = np.array(array)
    res = np.zeros_like(arr)
    curr = 0
    for i in range(res.size-1):
        curr += arr[i]*step # rectangles
        res[i+1] = curr
    return res

def integrale3(array, step):
    x = np.array(list(range(np.array(array).size)))/step
    res = [0]
    for i in range(1, x.size):
        y = array[:i]
        dx = x[:i]
        obj = integrate.simpson(y=y, x=dx)
        res.append(obj)
    result = [0]+[integrate.simpson(y=array[:i], x=x[:i]) for i in range(1,x.size)]
    return np.array(result)

def correlation(array1, array2):
    arr1, arr2 = np.ravel(np.array(array1)), np.ravel(np.array(array2))
    assert arr1.size == arr2.size, f"Cannot compute correlation coefficient with different size inputs: {arr1.size} != {arr2.size}"
    correlation_object = stats.pearsonr(arr1, arr2)
    return correlation_object.correlation



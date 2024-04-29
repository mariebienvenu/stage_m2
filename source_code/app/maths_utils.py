
import numpy as np
import scipy.stats as stats
from scipy.integrate import RK45 as runge_kutta
import scipy.integrate as integrate

def derivee(array, step:int|float):
    ar = np.array(array)
    res = (ar[2:]-ar[:-2])/(2*step) # différences finies
    return res


def derivee_seconde(array, step:int|float):
    ar = np.array(array)
    res = (-2*ar[1:-1] + ar[:-2] + ar[2:])/step**2  # différences finies
    return res


def integrale1(array, step:int|float):
    arr = np.array(array)
    res = np.zeros_like(arr)
    curr = 0
    for i in range(res.size-1):
        curr += arr[i]*step # rectangles
        res[i+1] = curr
    return res

def integrale3(array, step:int|float):
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


def dynamic_time_warping(array1, array2, debug=False):
    arr1, arr2 = np.ravel(np.array(array1)), np.ravel(np.array(array2))
    n,m = arr1.size, arr2.size
    cost_matrix = np.array([[abs(arr1[i] - arr2[j]) for j in range(m)] for i in range(n)]) # distance
    DTW = np.ones((n+1,m+1))*np.inf
    DTW[0,0] = 0
    for i in range(n):
        for j in range(m):
            cost = cost_matrix[i,j]
            additionnal_cost = min(DTW[i+1,j], DTW[i,j+1], DTW[i, j])
            DTW[i+1,j+1] = cost + additionnal_cost
    pairings = [[n-1,m-1]]
    i,j = n,m
    while i>1 or j>1:
        current = DTW[i,j]
        if DTW[i-1, j-1] <= current:
            i -= 1
            j -= 1
        elif DTW[i, j-1] <= current:
            j -= 1
        elif DTW[i-1, j] <= current:
            i -= 1
        pairings.append([i-1,j-1])
    return (DTW[n,m], pairings) if not debug else (DTW[n,m], pairings, DTW, cost_matrix)
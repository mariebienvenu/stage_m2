import numpy as np

from app.maths_utils import correlation
from random import random

arr1 = np.array([1,2,3])
arr2 = np.array([2,4,6])
corr = correlation(arr1, arr2)
print(corr)

arr1 = np.arange(0, 10, 0.1) # size 100
arr2 = arr1*(-0.5)
arr3 = np.exp(arr1)
arr4 = np.random.random(arr1.size) # random

print(correlation(arr1, arr2), correlation(arr1, arr3), correlation(arr1, arr4)) # expect -1, something, 0

arr5 = random()*arr1 + random()
print(correlation(arr1, arr5)) # expect 1

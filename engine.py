'''
Implementing logic for moving 1d, 2d and 3d walker
'''

import numpy as np
from scipy.stats import Normal

def gen_walk(n_steps, dimension, step_choices):
    walk = np.zeros([n_steps, dimension])
    n = 1
    while n < n_steps:
        axis = np.random.randint(dimension) #choose which axis to modify
        walk[n] = walk[n-1]
        walk[n][axis] = walk[n-1][axis] - np.random.choice(step_choices)
        n = n+1
    return walk

def rms_distance(step, step_length):
    return np.sqrt(step) * step_length


def confidence_interval(n_steps, dimension, step_length, confidence):
    if confidence < 0 or confidence > 1:
        raise ValueError("Confidence interval should be within 0 and 1")
    if dimension == 1:
        z = Normal(mu=0, sigma=1).icdf((1 + confidence) / 2) #the same as qnorm((1+confidence)/2) in R
        n = 1
        intervals = np.zeros([n_steps, 2])
        while n < n_steps:
            intervals[n][0] = -z * np.sqrt(n) * step_length
            intervals[n][1] =  z * np.sqrt(n)  * step_length #so we get [[lower boundary][upper boundary]]
            n = n + 1
        return intervals
        

'''
Implementing logic for moving 1d, 2d and 3d walker
'''

import numpy as np

def gen_walk(n_steps, dimension, step_choices):
    walk = np.zeros([n_steps, dimension])
    n = 1
    while n < n_steps:
        axis = np.random.randint(dimension) #choose which axis to modify
        walk[n] = walk[n-1]
        walk[n][axis] = walk[n-1][axis] - np.random.choice(step_choices)
        n = n+1
    return walk

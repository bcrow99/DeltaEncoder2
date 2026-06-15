'''
Created on Jun 7, 2026

@author: bcrow
'''
import numpy as np

'''
def get_histogram(src: np.ndarray)->np.ndarray:
    dim  = src.shape
    ydim = dim[0]
    xdim = dim[1]

    dst  = np.array([0] * 256)
    
    for i in range(0, ydim):
        for j in range(0, xdim): 
            k = src[i, j]
            dst[k] += 1
    
    return dst
    '''
    
def get_histogram(src: np.ndarray) -> np.ndarray:
    flat = src.ravel()
    hist = np.bincount(flat, minlength=256)
    return hist
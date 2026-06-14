'''
Created on Jun 9, 2026

@author: bcrow
'''

'''
def shift(src, shift):
    dim  = src.shape
    ydim = dim[0]
    xdim = dim[1]

    dst  = src.copy()

    
    if(shift < 0):
        for i in range(0, ydim):
            for j in range(0, xdim): 
                dst[i, j] >>= -shift
    else:
        for i in range(0, ydim):
            for j in range(0, xdim): 
                dst[i, j] <<= shift
    return dst
'''

def shift(src, shift):
    dst    = src.copy()
    factor = 1 << abs(shift)
    
    if(shift < 0):
        dst //= factor
    else:
        dst *= factor
    
    return dst
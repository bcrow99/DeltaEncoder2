import numpy as np

def shift(src, shift):
    dst    = src.copy()
    factor = 1 << abs(shift)
    
    if(shift < 0):
        dst //= factor
    else:
        dst *= factor
    
    return dst

def get_horizontal_deltas_from_values(src: np.ndarray) -> tuple:
    # Row-wise differences: each pixel minus its left neighbour.
    # np.diff gives columns 1..xdim-1; column 0 is always 0 (no left neighbour).
    ydim, xdim = src.shape
    dst = np.zeros((ydim, xdim), dtype=np.int16)
    dst[:, 1:] = np.diff(src, axis=1)

    # Column 0 of rows 1+ needs a special delta: each row's first pixel
    # minus the running cumulative sum of previous rows' column-0 deltas.
    # That running value is exactly src[i-1, 0] for each row i, which is
    # just shifting column 0 down by one row.
    dst[1:, 0] = np.diff(src[:, 0], axis=0)

    init_value = int(src[0, 0])
    return init_value, dst.ravel()


def get_values_from_horizontal_deltas(src: np.ndarray, xdim: int, ydim: int, init_value: int) -> np.ndarray:
    dst = src.reshape(ydim, xdim).astype(np.int32)

    # Recover column 0: cumsum of the inter-row deltas, starting from init_value.
    dst[0, 0] = init_value
    dst[:, 0] = np.cumsum(dst[:, 0])

    # Recover each row: cumsum across columns restores left-neighbour deltas.
    dst = np.cumsum(dst, axis=1)

    return dst.astype(np.uint8)


def get_horizontal_deltas_from_values2(src: np.ndarray) -> tuple:
    dim  = src.shape
    ydim = dim[0]
    xdim = dim[1]
    
    src = src.astype(np.int16)
    flat_src   = src.flatten();
    dst        = np.zeros((xdim * ydim), dtype=np.int16)
    init_value = flat_src[0]
    value      = init_value
    k = 0  

    for i in range(ydim):
        if i == 0:
            dst[k] = 0
            k += 1
        else:
            delta = flat_src[k] - init_value
            dst[k] = delta
            k += 1
            init_value += delta  
            value = init_value

        for j in range(1, xdim):  
            delta = flat_src[k] - value
            value += delta        
            dst[k] = delta
            k += 1

    return init_value, dst

def get_values_from_horizontal_deltas2(src: np.ndarray, xdim: int, ydim: int, init_value: int) -> np.ndarray:
    dst = np.zeros((ydim * xdim), dtype=np.uint8)
    k = 0
    value = init_value  

    for i in range(ydim):  
        if i != 0:
            value += src[k]
        current_value = value
        dst[k] = current_value  
        k += 1
        for j in range(1, xdim):  
            current_value += src[k]  
            dst[k] = current_value
            k += 1
    dst = dst.reshape(ydim, xdim)
    return dst

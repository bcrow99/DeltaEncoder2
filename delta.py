import numpy as np

def shift(src, shift):
    dst    = src.copy()
    factor = 1 << abs(shift)
    
    if(shift < 0):
        dst //= factor
    else:
        dst *= factor
    
    return dst

'''
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
    return init_value, dst
'''

def get_horizontal_deltas_from_values(src: np.ndarray) -> tuple:
    src = src.astype(np.int16)           # cast BEFORE np.diff sees it
    ydim, xdim = src.shape
    dst = np.zeros((ydim, xdim), dtype=np.int16)
    dst[:, 1:] = np.diff(src, axis=1)
    dst[1:, 0] = np.diff(src[:, 0])
    init_value = int(src[0, 0])
    return init_value, dst

def get_values_from_horizontal_deltas(src: np.ndarray, init_value: int) -> np.ndarray:
    dst = src.astype(np.int32)

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

def get_average_deltas_from_values(src: np.ndarray) -> tuple:
    src = src.astype(np.int16)
    ydim, xdim = src.shape
    dst = np.zeros((ydim, xdim), dtype=np.int16)

    dst[0, 1:] = np.diff(src[0])
    dst[1:, 0] = np.diff(src[:, 0])
    dst[1:, 1:] = src[1:, 1:] - ((src[1:, :-1].astype(np.int32) + src[:-1, 1:].astype(np.int32)) // 2).astype(np.int16)

    init_value = int(src[0, 0])
    return init_value, dst.ravel()


def get_values_from_average_deltas(src: np.ndarray, init_value: int) -> np.ndarray:
    dim = src.shape
    ydim = dim[0]
    xdim = dim[1]
    src = src.reshape(ydim, xdim).astype(np.int32)
    dst = np.zeros((ydim, xdim), dtype=np.int32)
    dst[0, 0] = init_value
    dst[0, 1:] = init_value + np.cumsum(src[0, 1:])

    for i in range(1, ydim):
        dst[i, 0] = dst[i - 1, 0] + src[i, 0]
        for j in range(1, xdim):
            dst[i, j] = (dst[i, j - 1] + dst[i - 1, j]) // 2 + src[i, j]

    return dst.astype(np.uint8).ravel()


def get_paeth_deltas_from_values(src: np.ndarray) -> tuple:
    src = src.astype(np.int16)
    ydim, xdim = src.shape
    dst = np.zeros((ydim, xdim), dtype=np.int16)

    dst[0, 1:] = np.diff(src[0])

    for i in range(1, ydim):
        dst[i, 0] = src[i, 0] - src[i - 1, 0]
        for j in range(1, xdim):
            a = src[i,     j - 1]
            b = src[i - 1, j    ]
            c = src[i - 1, j - 1]
            d = a + b - c

            if abs(a - d) <= abs(b - d) and abs(a - d) <= abs(c - d):
                dst[i, j] = src[i, j] - a
            elif abs(b - d) <= abs(c - d):
                dst[i, j] = src[i, j] - b
            else:
                dst[i, j] = src[i, j] - c

    init_value = int(src[0, 0])
    return init_value, dst.ravel()


def get_values_from_paeth_deltas(src: np.ndarray, init_value: int) -> np.ndarray:
    dim = src.shape
    ydim = dim[0]
    xdim = dim[1]
    src = src.reshape(ydim, xdim).astype(np.int32)
    dst = np.zeros((ydim, xdim), dtype=np.int32)
    dst[0, 0] = init_value
    dst[0, 1:] = init_value + np.cumsum(src[0, 1:])

    for i in range(1, ydim):
        dst[i, 0] = dst[i - 1, 0] + src[i, 0]
        for j in range(1, xdim):
            a = dst[i,     j - 1]
            b = dst[i - 1, j    ]
            c = dst[i - 1, j - 1]
            d = a + b - c

            if abs(a - d) <= abs(b - d) and abs(a - d) <= abs(c - d):
                dst[i, j] = a + src[i, j]
            elif abs(b - d) <= abs(c - d):
                dst[i, j] = b + src[i, j]
            else:
                dst[i, j] = c + src[i, j]

    return dst.astype(np.uint8).ravel()


def get_gradient_deltas_from_values(src: np.ndarray) -> tuple:
    src = src.astype(np.int16)
    ydim, xdim = src.shape
    dst = np.zeros((ydim, xdim), dtype=np.int16)

    dst[0, 1:] = np.diff(src[0])
    dst[1, 0]  = src[1, 0] - src[0, 0]
    dst[1, 1:] = np.diff(src[1])

    for i in range(2, ydim):
        dst[i, 0] = src[i, 0] - src[i - 1, 0]
        for j in range(1, xdim - 1):
            a = src[i,     j - 1]
            b = src[i - 1, j    ]
            c = src[i - 1, j - 1]
            d = src[i - 1, j + 1]
            e = src[i - 2, j - 1]

            gradients  = [abs(int(a) - int(e)), abs(int(c) - int(d)),
                          abs(int(a) - int(d)), abs(int(b) - int(e))]
            predictors = [src[i, j] - a, src[i, j] - b,
                          src[i, j] - c, src[i, j] - d]
            dst[i, j]  = predictors[np.argmax(gradients)]

        dst[i, xdim - 1] = src[i, xdim - 1] - src[i, xdim - 2]

    init_value = int(src[0, 0])
    return init_value, dst.ravel()


def get_values_from_gradient_deltas(src: np.ndarray, xdim: int, ydim: int, init_value: int) -> np.ndarray:
    src = src.reshape(ydim, xdim).astype(np.int32)
    dst = np.zeros((ydim, xdim), dtype=np.int32)

    dst[0, 0]  = init_value
    dst[0, 1:] = init_value + np.cumsum(src[0, 1:])
    dst[1, 0]  = dst[0, 0] + src[1, 0]
    dst[1, 1:] = dst[1, 0] + np.cumsum(src[1, 1:])
 
    ydim, xdim = src.shape
    for i in range(2, ydim):
        dst[i, 0] = dst[i - 1, 0] + src[i, 0]
        for j in range(1, xdim - 1):
            a = dst[i,     j - 1]
            b = dst[i - 1, j    ]
            c = dst[i - 1, j - 1]
            d = dst[i - 1, j + 1]
            e = dst[i - 2, j - 1]

            gradients  = [abs(a - e), abs(c - d), abs(a - d), abs(b - e)]
            predictors = [a, b, c, d]
            dst[i, j]  = predictors[np.argmax(gradients)] + src[i, j]

        dst[i, xdim - 1] = dst[i, xdim - 2] + src[i, xdim - 1]

    return dst.astype(np.uint8).ravel()
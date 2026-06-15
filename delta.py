import numpy as np

def shift(src, shift):
    dst    = src.copy()
    factor = 1 << abs(shift)
    
    if(shift < 0):
        dst //= factor
    else:
        dst *= factor
    
    return dst


def get_horizontal_deltas_from_values(src: np.ndarray, xdim: int, ydim: int) -> tuple:
    src2d = src.reshape(ydim, xdim)
    dst = np.zeros((ydim, xdim), dtype=np.int32)
    init_value = int(src[0])

    dst[:, 1:] = np.diff(src2d, axis=1)   # all rows: horizontal diffs

    sum_ = int(np.abs(dst).sum())
    return sum_, dst.ravel(), init_value


def get_values_from_horizontal_deltas(src: np.ndarray, xdim: int, ydim: int, init_value: int) -> np.ndarray:
    dst = src.reshape(ydim, xdim).copy()
    dst[0, 0] = init_value
    dst[:, 0] = np.cumsum(dst[:, 0])
    return np.cumsum(dst, axis=1).ravel()

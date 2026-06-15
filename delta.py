import numpy as np

def shift(src, shift):
    dst    = src.copy()
    factor = 1 << abs(shift)
    
    if(shift < 0):
        dst //= factor
    else:
        dst *= factor
    
    return dst

def get_vertical_deltas_from_values(src: np.ndarray, xdim: int, ydim: int) -> tuple:
    dst = np.zeros(xdim * ydim, dtype=np.int32)
    init_value = src[0]
    value = init_value
    sum_ = 0

    k = 0
    for i in range(ydim):
        for j in range(xdim):
            if i == 0:
                if j == 0:
                    dst[k] = 0
                    k += 1
                else:
                    delta = src[k] - value
                    value += delta
                    dst[k] = delta
                    k += 1
                    sum_ += abs(delta)
            else:
                delta = src[k] - src[k - xdim]
                dst[k] = delta
                k += 1
                sum_ += abs(delta)

    return sum_, dst, init_value


def get_values_from_vertical_deltas(src: np.ndarray, xdim: int, ydim: int, init_value: int) -> np.ndarray:
    dst = np.zeros(xdim * ydim, dtype=np.int32)
    dst[0] = init_value
    value = init_value

    for i in range(1, xdim):
        value += src[i]
        dst[i] = value

    for i in range(1, ydim):
        for j in range(xdim):
            index = i * xdim + j
            dst[index] = dst[index - xdim] + src[index]

    return dst

def get_vertical_deltas_from_values2(src: np.ndarray, xdim: int, ydim: int) -> tuple:

    dst = np.zeros(xdim * ydim, dtype=np.int32)

    init_value = src[0]

    value = init_value

    sum_ = 0



    k = 0

    for i in range(ydim):

        for j in range(xdim):

            if i == 0:

                if j == 0:

                    dst[k] = 0

                    k += 1

                else:

                    delta = src[k] - value

                    value += delta

                    dst[k] = delta

                    k += 1

                    sum_ += abs(delta)

            else:

                delta = src[k] - src[k - xdim]

                dst[k] = delta

                k += 1

                sum_ += abs(delta)



    return sum_, dst, init_value





def get_values_from_vertical_deltas2(src: np.ndarray, xdim: int, ydim: int, init_value: int) -> np.ndarray:

    dst = np.zeros(xdim * ydim, dtype=np.int32)

    dst[0] = init_value

    value = init_value



    for i in range(1, xdim):

        value += src[i]

        dst[i] = value



    for i in range(1, ydim):

        for j in range(xdim):

            index = i * xdim + j

            dst[index] = dst[index - xdim] + src[index]



    return dst
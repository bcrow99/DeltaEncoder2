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

def get_rank_table(src: list[int]) -> list[int]:
    """
    Creates a rank table of most popular to least popular from a histogram.

    :param src: the input histogram
    :return:    rank table
    """
    table = {}
    lst   = []
    for i, val in enumerate(src):
        key = float(val)
        while key in table:
            key += 0.001
        table[key] = i
        lst.append(key)

    lst.sort()

    rank = [0] * len(src)
    k    = -1
    for i in range(len(src) - 1, -1, -1):
        key     = lst[i]
        j       = table[key]
        k      += 1
        rank[j] = k

    return rank

def pack_strings(src: list[int], table: list[int]) -> tuple[int, bytearray]:
    """
    Uses a rank table of most popular to least popular to turn integer
    representations into unary strings, and then pack them into a byte
    array. Returns a tuple of (number of bits written, packed bytearray).

    :param src:   the input integers
    :param table: the rank table
    :return:      (number of bits written, packed bytearray)
    """
    max_bits = sum(table[src[i]] + 1 for i in range(len(src)))
    max_bytes = (max_bits + 7) // 8
    dst = bytearray(max_bytes)

    mask = [1, 3, 7, 15, 31, 63, 127, 255]

    start_bit = 0
    p = 0

    for i in range(len(src)):
        j = src[i]
        k = table[j]
        if k == 0:
            start_bit += 1
            if start_bit == 8:
                p += 1
                dst[p] = 0
                start_bit = 0
        else:
            stop_bit = (start_bit + k + 1) % 8

            if k <= 7:
                dst[p] = (dst[p] | (mask[k - 1] << start_bit)) & 0xFF
                if stop_bit <= start_bit:
                    p += 1
                    dst[p] = 0
                    if stop_bit != 0:
                        bits_in_next = k - (8 - start_bit)
                        dst[p] = (dst[p] | mask[bits_in_next - 1]) & 0xFF
            else:  # k > 7
                dst[p] = (dst[p] | (mask[7] << start_bit)) & 0xFF
                m = (k - 8) // 8
                for n in range(m):
                    p += 1
                    dst[p] = mask[7]
                p += 1
                dst[p] = 0
                if start_bit != 0:
                    dst[p] = (dst[p] | (mask[7] >> (8 - start_bit))) & 0xFF

                if k % 8 != 0:
                    m = k % 8 - 1
                    dst[p] = (dst[p] | (mask[m] << start_bit)) & 0xFF
                    if stop_bit <= start_bit:
                        p += 1
                        dst[p] = 0
                        if stop_bit != 0:
                            dst[p] = (dst[p] | mask[stop_bit - 1]) & 0xFF
                elif stop_bit <= start_bit:
                    p += 1
                    dst[p] = 0

            start_bit = stop_bit

    if start_bit != 0:
        p += 1
    number_of_bits = p * 8
    if start_bit != 0:
        number_of_bits -= 8 - start_bit

    return number_of_bits, dst[:p if start_bit == 0 else p]


def unpack_strings(src: bytes | bytearray, table: list[int], size: int) -> tuple[int, list[int]]:
    """
    Uses a rank table of most popular to least popular to turn unary strings
    into integers.

    :param src:   the input bytes
    :param table: the rank table
    :param size:  number of values to unpack
    :return:      (number of values unpacked, list of unpacked integers)
    """
    dst = [0] * size
    number_of_different_values = len(table)
    number_unpacked = 0

    inverse_table = [0] * number_of_different_values
    for i in range(number_of_different_values):
        j = table[i]
        inverse_table[j] = i

    length = 0
    src_byte = 0
    dst_byte = 0
    bit = 0

    while dst_byte < size:
        non_zero = src[src_byte] & (0x01 << bit)
        if non_zero != 0:
            length += 1
        else:
            k = length
            dst[dst_byte] = inverse_table[k]
            dst_byte += 1
            number_unpacked += 1
            length = 0
        bit += 1
        if bit == 8:
            bit = 0
            src_byte += 1

    return number_unpacked, dst
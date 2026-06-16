'''
Created on Jun 7, 2026

@author: bcrow
'''
import numpy as np


def get_histogram(src: np.ndarray) -> list[int]:
    flat = src.ravel()
    hist = np.bincount(flat, minlength=256)
    return hist.tolist()

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


def pack_strings(src: np.ndarray, table: list[int]) -> tuple[int, bytearray]:
    """
    Uses a rank table of most popular to least popular to turn integer
    representations into unary strings, and then pack them into a byte
    array. Returns a tuple of (number of bits written, packed bytearray).

    :param src:   2D input array of integers
    :param table: the rank table
    :return:      (number of bits written, packed bytearray)
    """
    flat = src.ravel()
    number_of_values = len(table)
    maximum_length = number_of_values - 1

    #max_bits = sum(table[v] + 1 for v in flat)
    max_bytes = (len(flat) * (maximum_length + 1) + 7) // 8 + 1
    dst = bytearray(max_bytes)

    mask = [1, 3, 7, 15, 31, 63, 127, 255]

    start_bit = 0
    stop_bit = 0
    p = 0

    for v in flat:
        k = table[v]
        if k == 0:
            start_bit += 1
            if start_bit == 8:
                p += 1
                dst[p] = 0
                start_bit = 0
        else:
            stop_bit = (start_bit + k + 1) % 8
            if k == maximum_length:
                stop_bit -= 1
                if stop_bit < 0:
                    stop_bit = 7

            if k <= 7:
                dst[p] = (dst[p] | (mask[k - 1] << start_bit)) & 0xFF
                if stop_bit <= start_bit:
                    p += 1
                    dst[p] = 0
                    if stop_bit != 0:
                        dst[p] = (dst[p] | (mask[k - 1] >> (8 - start_bit))) & 0xFF
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
                            dst[p] = (dst[p] | (mask[m] >> (8 - start_bit))) & 0xFF
                elif stop_bit <= start_bit and k != maximum_length:
                    p += 1
                    dst[p] = 0

            start_bit = stop_bit

    if start_bit != 0:
        p += 1
    number_of_bits = p * 8
    if start_bit != 0:
        number_of_bits -= 8 - start_bit

    return number_of_bits, dst[:p] if start_bit == 0 else dst[:p + 1]


def unpack_strings(src: bytes | bytearray, table: list[int], shape: tuple[int, int], number_of_bits: int) -> tuple[int, np.ndarray]:
    """
    Uses a rank table of most popular to least popular to turn unary strings
    into integers, returning a 2D array of the given shape.

    :param src:           the input bytes
    :param table:         the rank table
    :param shape:         (rows, cols) of the output array
    :param number_of_bits: number of bits written by pack_strings
    :return:              (number of values unpacked, 2D array of unpacked integers)
    """
    rows, cols = shape
    size = rows * cols
    number_of_values = len(table)
    maximum_length = number_of_values - 1
    flat = [0] * size
    number_unpacked = 0

    inverse_table = [0] * number_of_values
    for i in range(number_of_values):
        inverse_table[table[i]] = i

    length = 1
    src_byte = 0
    dst_byte = 0
    bit = 0
    bits_read = 0

    try:
        while dst_byte < size and bits_read < number_of_bits:
            non_zero = src[src_byte] & (1 << bit)
            if non_zero != 0 and length < maximum_length:
                length += 1
            elif non_zero == 0:
                k = length - 1
                flat[dst_byte] = inverse_table[k]
                dst_byte += 1
                number_unpacked += 1
                length = 1
            elif length == maximum_length:
                k = length
                flat[dst_byte] = inverse_table[k]
                dst_byte += 1
                number_unpacked += 1
                length = 1
            bit += 1
            bits_read += 1
            if bit == 8:
                bit = 0
                src_byte += 1
    except Exception as e:
        print(f"{e}")
        print("Exiting unpack_strings with an exception.")

    dst = np.array(flat, dtype=np.int16).reshape(shape)
    return number_unpacked, dst
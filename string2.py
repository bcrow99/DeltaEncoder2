import numpy as np


# Low level bit manipulation utilities

def get_bit_table() -> list[int]:
    """Creates a table containing the number of 0 bits in any byte value."""
    return [sum(1 for j in range(8) if (i & (1 << j)) == 0) for i in range(256)]


def get_zero_ratio(string: bytes | bytearray, bit_length: int, bit_table: list[int]) -> float:
    """Returns the ratio of zero bits to total bits in the string."""
    byte_length = bit_length // 8
    zero_sum = sum(bit_table[string[i] & 0xFF] for i in range(byte_length))
    one_sum  = byte_length * 8 - zero_sum

    for i in range(bit_length % 8):
        if (string[byte_length] & (1 << i)) == 0:
            zero_sum += 1
        else:
            one_sum += 1

    return zero_sum / (zero_sum + one_sum)


def get_compression_amount(string: bytes | bytearray, bit_length: int, transform_type: int) -> int:
    """
    Returns the expected change in bit length after one compression iteration.
    Negative means compression will reduce size, positive means it will expand.
    transform_type 0 = zero bit compression, transform_type 1 = one bit compression.
    """
    positive = 0
    negative = 0
    mask = 1
    byte_length = bit_length // 8
    previous = 1 if transform_type == 0 else 0

    def process_bit(bit_val):
        nonlocal positive, negative, previous
        if transform_type == 0:
            if bit_val != 0 and previous != 0:
                positive += 1
            elif bit_val != 0 and previous == 0:
                previous = 1
            elif bit_val == 0 and previous != 0:
                previous = 0
            else:
                negative += 1
                previous = 1
        else:
            if bit_val == 0 and previous == 0:
                positive += 1
            elif bit_val == 0 and previous == 1:
                previous = 0
            elif bit_val != 0 and previous == 0:
                previous = 1
            else:
                negative += 1
                previous = 0

    for i in range(byte_length):
        for j in range(8):
            process_bit(string[i] & (mask << j))

    for i in range(bit_length % 8):
        process_bit(string[byte_length] & (mask << i))

    return positive - negative


# Trailing data byte functions

def get_bitlength(string: bytes | bytearray) -> int:
    """Extracts the bit length from the trailing data byte."""
    return (len(string) - 1) * 8 - ((string[-1] >> 5) & 7)


def get_bytelength(bitlength: int) -> int:
    """Returns number of bytes needed to store bitlength bits, plus one data byte."""
    return (bitlength + 7) // 8 + 1


def get_iterations(string: bytes | bytearray) -> int:
    """Extracts the iteration count from the trailing data byte."""
    return string[-1] & 31


def set_iterations(iterations: int, string: bytearray) -> None:
    """Sets the iteration count in the trailing data byte."""
    string[-1] = (string[-1] & 0b11100000) | (iterations & 0x1F)


def get_type(string: bytes | bytearray) -> int:
    """Returns 0 for zero-type string, 1 for one-type string."""
    return 1 if (string[-1] & 31) > 15 else 0


def set_data(type: int, iterations: int, bitlength: int, string: bytearray) -> None:
    """Sets the trailing data byte encoding type, iterations, and bit length."""
    if type == 1:
        iterations += 16
    odd_bits = bitlength % 8
    extra_bits = ((8 - odd_bits) << 5) & 0xFF if odd_bits != 0 else 0
    string[-1] = (iterations & 0xFF) | extra_bits


def print_bits(string: bytes | bytearray, bitlength: int) -> None:
    """Prints the bits of a string in MSB-first order."""
    bytelength = (bitlength + 7) // 8
    result = []
    i = bytelength - 1
    if bitlength % 8 != 0:
        for j in range(bitlength % 8 - 1, -1, -1):
            result.append('1' if (string[i] & (1 << j)) != 0 else '0')
        result.append(' ')
        i -= 1
    while i >= 0:
        for j in range(7, -1, -1):
            result.append('1' if (string[i] & (1 << j)) != 0 else '0')
        result.append(' ')
        i -= 1
    print(''.join(result))


# Histogram and rank table functions

def get_histogram(src: np.ndarray) -> list[int]:
    """
    Returns a histogram of byte values in src.

    :param src: input array
    :return:    list of 256 counts
    """
    return np.bincount(src.ravel(), minlength=256).tolist()


def get_rank_table(histogram: list[int]) -> list[int]:
    """
    Builds a rank table mapping each value to its popularity rank,
    most popular = 0.

    :param histogram: list of 256 counts
    :return:          rank table
    """
    table = [0] * len(histogram)
    for rank, value in enumerate(sorted(range(len(histogram)), key=lambda i: histogram[i], reverse=True)):
        table[value] = rank
    return table


# Core pack and unpack functions

def pack_strings(src: np.ndarray, table: list[int]) -> tuple[int, bytearray]:
    """
    Uses a rank table of most popular to least popular to turn integer
    representations into unary strings, and then pack them into a byte array.

    :param src:   2D input array of integers
    :param table: the rank table
    :return:      (number of bits written, packed bytearray)
    """
    flat = src.ravel()
    number_of_values = len(table)
    maximum_length = number_of_values - 1

    dst = bytearray((len(flat) * (maximum_length + 1) + 7) // 8 + 1)
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
                start_bit = 0
        else:
            stop_bit = (start_bit + k + 1) % 8
            if k == maximum_length:
                stop_bit = (stop_bit - 1) % 8 if stop_bit > 0 else 7

            if k <= 7:
                dst[p] = (dst[p] | (mask[k - 1] << start_bit)) & 0xFF
                if stop_bit <= start_bit:
                    p += 1
                    if stop_bit != 0:
                        dst[p] = (dst[p] | (mask[k - 1] >> (8 - start_bit))) & 0xFF
            else:
                dst[p] = (dst[p] | (mask[7] << start_bit)) & 0xFF
                m = (k - 8) // 8
                for n in range(m):
                    p += 1
                    dst[p] = mask[7]
                p += 1
                if start_bit != 0:
                    dst[p] = (dst[p] | (mask[7] >> (8 - start_bit))) & 0xFF
                if k % 8 != 0:
                    m = k % 8 - 1
                    dst[p] = (dst[p] | (mask[m] << start_bit)) & 0xFF
                    if stop_bit <= start_bit:
                        p += 1
                        if stop_bit != 0:
                            dst[p] = (dst[p] | (mask[m] >> (8 - start_bit))) & 0xFF
                elif stop_bit <= start_bit and k != maximum_length:
                    p += 1

            start_bit = stop_bit

    if start_bit != 0:
        p += 1
    number_of_bits = p * 8 - (8 - start_bit if start_bit != 0 else 0)
    return number_of_bits, dst[:p] if start_bit == 0 else dst[:p + 1]


def unpack_strings(src: bytes | bytearray, table: list[int], shape: tuple[int, int], number_of_bits: int) -> tuple[int, np.ndarray]:
    """
    Uses a rank table of most popular to least popular to turn unary strings
    into integers, returning a 2D array of the given shape.

    :param src:            the input bytes
    :param table:          the rank table
    :param shape:          (rows, cols) of the output array
    :param number_of_bits: number of bits written by pack_strings
    :return:               (number of values unpacked, 2D array of unpacked integers)
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
                flat[dst_byte] = inverse_table[length - 1]
                dst_byte += 1
                number_unpacked += 1
                length = 1
            elif length == maximum_length:
                flat[dst_byte] = inverse_table[length]
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

    return number_unpacked, np.array(flat, dtype=np.int32).reshape(shape)


# Low level bit transform functions

def _advance(current_byte: int, current_bit: int) -> tuple[int, int]:
    """Advances the bit cursor by one, wrapping to the next byte if needed."""
    current_bit += 1
    if current_bit == 8:
        current_byte += 1
        current_bit = 0
    return current_byte, current_bit


def _advance_safe(current_byte: int, current_bit: int, dst_len: int) -> tuple[int, int]:
    """Advances the bit cursor by one, only resetting current_bit if within bounds."""
    current_bit += 1
    if current_bit == 8:
        current_byte += 1
        if current_byte < dst_len:
            current_bit = 0
    return current_byte, current_bit


def _write_bit(dst: bytearray, current_byte: int, current_bit: int) -> tuple[int, int]:
    """Writes a 1 bit at the current position and advances the cursor."""
    dst[current_byte] = (dst[current_byte] | (0x01 << current_bit)) & 0xFF
    return _advance(current_byte, current_bit)


def _write_bit_safe(dst: bytearray, current_byte: int, current_bit: int) -> tuple[int, int]:
    """Writes a 1 bit at the current position and advances safely."""
    dst[current_byte] = (dst[current_byte] | (0x01 << current_bit)) & 0xFF
    return _advance_safe(current_byte, current_bit, len(dst))


def compress_zero_bits(src: bytes | bytearray, size: int, dst: bytearray) -> list[int]:
    """
    Bitwise substitution that compresses zero bits:
      00 -> 0, 01 -> 11, 1x -> 1 (skip next), odd trailing 0 -> 1
    Returns [new_bit_length, odd_flag].
    """
    result = [0, 0]
    for i in range(len(dst)):
        dst[i] = 0

    cb, cbit = 0, 0
    i, j, k = 0, 0, 0

    try:
        while i < size:
            if (src[k] & (0x01 << j)) == 0 and i < size - 1:
                i += 1
                j += 1
                if j == 8:
                    j, k = 0, k + 1
                if (src[k] & (0x01 << j)) == 0:
                    # 00 -> 0
                    cb, cbit = _advance(cb, cbit)
                else:
                    # 01 -> 11
                    cb, cbit = _write_bit(dst, cb, cbit)
                    cb, cbit = _write_bit(dst, cb, cbit)
            elif (src[k] & (0x01 << j)) == 0 and i == size - 1:
                # Odd trailing 0 -> 1
                cb, cbit = _write_bit(dst, cb, cbit)
                result[1] = 1
            else:
                # 1x -> 1, skip next
                cb, cbit = _write_bit(dst, cb, cbit)
                cb, cbit = _advance(cb, cbit)
            i += 1
            j += 1
            if j == 8:
                j, k = 0, k + 1
    except Exception as e:
        print(f"{e}")
        print("Exiting compress_zero_bits with an exception.")

    result[0] = cb * 8 + cbit
    return result


def decompress_zero_bits(src: bytes | bytearray, size: int, dst: bytearray) -> int:
    """
    Inverse of compress_zero_bits:
      0 -> 00, 11 -> 01, lone 1 -> append odd 0
    Returns new bit length.
    """
    for i in range(len(dst)):
        dst[i] = 0

    cb, cbit = 0, 0
    i, j, k = 0, 0, 0

    try:
        while i < size:
            if (src[k] & (0x01 << j)) != 0 and i < size - 1:
                i += 1
                j += 1
                if j == 8:
                    j, k = 0, k + 1
                if (src[k] & (0x01 << j)) != 0:
                    # 11 -> 01
                    cb, cbit = _advance(cb, cbit)
                    cb, cbit = _write_bit(dst, cb, cbit)
                else:
                    # 10 -> 1
                    cb, cbit = _write_bit(dst, cb, cbit)
            elif (src[k] & (0x01 << j)) != 0 and i == size - 1:
                # Lone 1 -> append odd 0
                cb, cbit = _advance_safe(cb, cbit, len(dst))
            else:
                # 0 -> 00
                cb, cbit = _advance_safe(cb, cbit, len(dst))
                cb, cbit = _advance_safe(cb, cbit, len(dst))
            i += 1
            j += 1
            if j == 8:
                j, k = 0, k + 1
    except Exception as e:
        print(f"{e}")
        print(f"Input size was {size}")
        print("Exiting decompress_zero_bits with an exception.")

    return cb * 8 + cbit


def compress_one_bits(src: bytes | bytearray, size: int, dst: bytearray) -> list[int]:
    """
    Bitwise substitution that compresses one bits:
      11 -> 1, 10 -> 01, 0x -> 00 (skip next), odd trailing 1 -> 0
    Returns [new_bit_length, odd_flag].
    """
    result = [0, 0]
    for i in range(len(dst)):
        dst[i] = 0

    cb, cbit = 0, 0
    i, j, k = 0, 0, 0

    while i < size:
        if (src[k] & (0x01 << j)) != 0 and i < size - 1:
            i += 1
            j += 1
            if j == 8:
                j, k = 0, k + 1
            if (src[k] & (0x01 << j)) != 0:
                # 11 -> 1
                cb, cbit = _write_bit(dst, cb, cbit)
            else:
                # 10 -> 01
                cb, cbit = _advance(cb, cbit)
                cb, cbit = _write_bit(dst, cb, cbit)
        elif (src[k] & (0x01 << j)) != 0 and i == size - 1:
            # Odd trailing 1 -> 0
            cb, cbit = _advance(cb, cbit)
            result[1] = 1
        else:
            # 0x -> 00
            cb, cbit = _advance(cb, cbit)
            cb, cbit = _advance(cb, cbit)
        i += 1
        j += 1
        if j == 8:
            j, k = 0, k + 1

    result[0] = cb * 8 + cbit
    return result


def decompress_one_bits(src: bytes | bytearray, size: int, dst: bytearray) -> int:
    """
    Inverse of compress_one_bits:
      1 -> 11, 01 -> 10, 00 -> 0, lone 0 -> append odd 1
    Returns new bit length.
    """
    for i in range(len(dst)):
        dst[i] = 0

    cb, cbit = 0, 0
    i, j, k = 0, 0, 0

    while i < size:
        try:
            if (src[k] & (0x01 << j)) == 0 and i < size - 1:
                i += 1
                j += 1
                if j == 8:
                    j, k = 0, k + 1
                if (src[k] & (0x01 << j)) == 0:
                    # 00 -> 0
                    cb, cbit = _advance(cb, cbit)
                else:
                    # 01 -> 10
                    cb, cbit = _write_bit_safe(dst, cb, cbit)
                    cb, cbit = _advance_safe(cb, cbit, len(dst))
            elif (src[k] & (0x01 << j)) == 0 and i == size - 1:
                # Lone 0 -> append odd 1
                cb, cbit = _write_bit_safe(dst, cb, cbit)
            else:
                # 1 -> 11
                cb, cbit = _write_bit_safe(dst, cb, cbit)
                cb, cbit = _write_bit_safe(dst, cb, cbit)
            i += 1
            j += 1
            if j == 8:
                j, k = 0, k + 1
        except Exception as e:
            print(f"{e}")
            print(f"Size of input was {size}")
            print(f"Exception on index {i}")
            print("Exiting decompress_one_bits.")
            break

    return cb * 8 + cbit


# Iterative compress and decompress functions

def _compress_strings(src: bytes | bytearray, bitlength: int, transform_type: int) -> bytearray:
    """
    Iteratively applies compress_zero_bits or compress_one_bits until
    compression is achieved or limit is reached.
    """
    compress_bits = compress_zero_bits if transform_type == 0 else compress_one_bits
    limit = 15 if transform_type == 0 else 16

    try:
        amount = get_compression_amount(src, bitlength, transform_type)
        if amount > 0:
            bytelength = get_bytelength(bitlength)
            dst = bytearray(bytelength)
            dst[:bytelength - 1] = src[:bytelength - 1]
            set_data(transform_type, 0, bitlength, dst)
            return dst

        buffer1 = bytearray(len(src))
        buffer2 = bytearray(len(src))

        result = compress_bits(src, bitlength, buffer1)
        compressed_length = result[0]
        amount = get_compression_amount(buffer1, compressed_length, transform_type)
        iterations = 1

        while amount < 0 and iterations < limit:
            previous_length = compressed_length
            if iterations % 2 == 1:
                result = compress_bits(buffer1, previous_length, buffer2)
                compressed_length = result[0]
                amount = get_compression_amount(buffer2, compressed_length, transform_type)
            else:
                result = compress_bits(buffer2, previous_length, buffer1)
                compressed_length = result[0]
                amount = get_compression_amount(buffer1, compressed_length, transform_type)
            iterations += 1

        bytelength = get_bytelength(compressed_length)
        dst = bytearray(bytelength)
        final_buffer = buffer2 if iterations % 2 == 0 else buffer1
        dst[:bytelength - 1] = final_buffer[:bytelength - 1]
        set_data(transform_type, iterations, compressed_length, dst)
        return dst

    except Exception as e:
        print(f"{e}")
        print(f"Exiting _compress_strings (type={transform_type}) with an exception.")
        return bytearray(1)


def _decompress_strings(src: bytes | bytearray) -> bytearray:
    """
    Iteratively decompresses a string using the trailing data byte to
    determine type and number of iterations.
    """
    decompress_bits = decompress_zero_bits if get_type(src) == 0 else decompress_one_bits
    transform_type = get_type(src)
    iterations = get_iterations(src)
    bitlength = get_bitlength(src)

    if transform_type == 0:
        if iterations == 0:
            return bytearray(src)
    else:
        if iterations <= 16:
            return bytearray(src)
        iterations -= 16

    original_iterations = iterations

    bytelength = int(get_bytelength(bitlength) * (2 ** iterations))
    buffer1 = bytearray(bytelength)
    buffer2 = bytearray(bytelength)

    uncompressed_length = decompress_bits(src, bitlength, buffer1)
    iterations -= 1

    while iterations > 0:
        previous_length = uncompressed_length
        if iterations % 2 == 1:
            uncompressed_length = decompress_bits(buffer1, previous_length, buffer2)
        else:
            uncompressed_length = decompress_bits(buffer2, previous_length, buffer1)
        iterations -= 1

    bytelength = get_bytelength(uncompressed_length)
    dst = bytearray(bytelength)
    final_buffer = buffer2 if original_iterations % 2 == 0 else buffer1
    dst[:bytelength - 1] = final_buffer[:bytelength - 1]
    set_data(transform_type, 0, uncompressed_length, dst)
    return dst


def compress_strings(string: bytes | bytearray) -> bytearray:
    """Compresses a string using either zero or one bit compression based on type."""
    return _compress_strings(string, get_bitlength(string), get_type(string))


def decompress_strings(string: bytes | bytearray) -> bytearray:
    """Decompresses a string using either zero or one bit decompression based on type."""
    iterations = get_iterations(string)
    if iterations == 0 or iterations == 16:
        return bytearray(string)
    return _decompress_strings(string)


# High level pack/unpack with compression

def pack_and_compress(src: np.ndarray, table: list[int]) -> bytearray:
    """
    Packs a 2D array of integers into unary strings, then compresses the
    result. Attaches a trailing data byte so the compress functions work
    correctly.

    :param src:   2D input array of integers
    :param table: the rank table
    :return:      compressed bytearray with trailing data byte
    """
    bitlength, packed = pack_strings(src, table)
    bytelength = get_bytelength(bitlength)
    dst = bytearray(bytelength)
    dst[:bytelength - 1] = packed[:bytelength - 1]
    set_data(0, 0, bitlength, dst)
    return compress_strings(dst)


def decompress_and_unpack(src: bytearray, table: list[int], shape: tuple[int, int]) -> tuple[int, np.ndarray]:
    """
    Decompresses a compressed bytearray and unpacks it into a 2D array of
    integers. Strips the trailing data byte before passing to unpack_strings.

    :param src:   compressed bytearray with trailing data byte
    :param table: the rank table
    :param shape: (rows, cols) of the output array
    :return:      (number of values unpacked, 2D array of unpacked integers)
    """
    decompressed = decompress_strings(src)
    bitlength = get_bitlength(decompressed)
    raw = bytes(decompressed[:get_bytelength(bitlength) - 1])
    return unpack_strings(raw, table, shape, bitlength)
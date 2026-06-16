from math import gcd


def get_arithmetic_offset(src: bytes, frequency: list[int]) -> tuple[int, int]:
    """
    Arithmetic encode `src` using the given symbol frequencies.

    Args:
        src:       Raw bytes to encode.
        frequency: frequency[i] = how many times byte value i appears in src.

    Returns:
        (numerator, denominator) representing the encoded rational offset.
    """
    f = frequency[:]
    n = len(src)

    # Build cumulative-frequency (start) table
    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    # offset and range are stored as (numerator, denominator) pairs
    offset = [0, 1]   # 0/1
    range_ = [1, 1]   # 1/1

    for i in range(n):
        j = src[i]  # Python bytes iteration already gives int 0-255

        # addend = range * s[j] / m
        addend = [range_[0] * s[j], range_[1] * m]
        g = gcd(addend[0], addend[1])
        if g > 1:
            addend[0] //= g
            addend[1] //= g

        # offset += addend  (rational addition)
        offset[0] = offset[0] * addend[1] + addend[0] * offset[1]
        offset[1] = offset[1] * addend[1]

        g = gcd(offset[0], offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        # range *= f[j] / m
        range_[0] = range_[0] * f[j]
        range_[1] = range_[1] * m

        g = gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        # Adaptive: remove symbol j from the model
        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    return (offset[0], offset[1])


def get_arithmetic_values(v: tuple[int, int], frequency: list[int], n: int) -> bytes:
    """
    Arithmetic decode `n` symbols from the rational offset `v`.

    Args:
        v:         (numerator, denominator) returned by get_arithmetic_offset.
        frequency: Same frequency table used during encoding.
        n:         Number of symbols to decode.

    Returns:
        Decoded bytes.
    """
    value = bytearray(n)

    # Build list of [symbol, freq, cumulative_start] for non-zero symbols
    arith_list = []
    m = 0
    for i in range(len(frequency)):
        if frequency[i] != 0:
            arith_list.append([i, frequency[i], m])
            m += frequency[i]

    offset = [0, 1]
    range_ = [1, 1]
    w = [v[0], v[1]]

    for i in range(n):
        # w = v - offset  (only recompute when offset is non-zero)
        if offset[0] != 0:
            w[0] = v[0] * offset[1] - offset[0] * v[1]
            w[1] = v[1] * offset[1]

            g = gcd(abs(w[0]), w[1])
            if g > 1:
                w[0] //= g
                w[1] //= g

        # Binary-search arith_list for the symbol whose interval contains w
        j = len(arith_list) // 2
        entry = arith_list[j]
        f_j = entry[1]
        s_j = entry[2]

        # Interval check:  range*s/m <= w < range*(s+f)/m
        # Multiply through by w[1]*d to stay in integers:
        #   a = range[0]*s   * w[1]
        #   b = w[0]         * range[1]*m
        #   c = range[0]*(s+f) * w[1]
        def make_abc(s_val, f_val):
            d = range_[1] * m
            a = range_[0] * s_val * w[1]
            b = w[0] * d
            c = range_[0] * (s_val + f_val) * w[1]
            return a, b, c

        a, b, c = make_abc(s_j, f_j)

        if a > b:
            # Search left
            k = max(j // 2, 1)
            while a > b:
                j -= k
                entry = arith_list[j]
                f_j, s_j = entry[1], entry[2]
                a = range_[0] * s_j * w[1]
                k = max(k // 2, 1)

            # Check if we overshot (c <= b means w is past this symbol's interval)
            c = range_[0] * (s_j + f_j) * w[1]
            if c <= b:
                while c <= b:
                    j += 1
                    entry = arith_list[j]
                    f_j, s_j = entry[1], entry[2]
                    c = range_[0] * (s_j + f_j) * w[1]

        elif c <= b:
            # Search right
            k = max((len(arith_list) - j) // 2, 1)
            while c <= b:
                j += k
                entry = arith_list[j]
                f_j, s_j = entry[1], entry[2]
                c = range_[0] * (s_j + f_j) * w[1]
                k = max(k // 2, 1)

            # Check if we overshot (a > b means w is before this symbol's interval)
            a = range_[0] * s_j * w[1]
            if a > b:
                while a > b:
                    j -= 1
                    entry = arith_list[j]
                    f_j, s_j = entry[1], entry[2]
                    a = range_[0] * s_j * w[1]

        # Update offset: offset += range * s_j / m
        addend_num = range_[0] * s_j
        addend_den = range_[1] * m

        offset[0] = offset[0] * addend_den + addend_num * offset[1]
        offset[1] = offset[1] * addend_den

        g = gcd(abs(offset[0]), offset[1])
        if g > 1:
            offset[0] //= g
            offset[1] //= g

        # Update range: range *= f_j / m
        range_[0] = range_[0] * f_j
        range_[1] = range_[1] * m

        g = gcd(range_[0], range_[1])
        if g > 1:
            range_[0] //= g
            range_[1] //= g

        # Adaptive: decrement cumulative starts for all symbols after j
        for p in range(j + 1, len(arith_list)):
            arith_list[p][2] -= 1

        f_j -= 1
        m -= 1

        if f_j != 0:
            arith_list[j][1] = f_j
        else:
            arith_list.pop(j)

        value[i] = entry[0]

    return bytes(value)


# ---------------------------------------------------------------------------
# Quick round-trip smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    data = b"hello arithmetic world"

    # Build frequency table (256 buckets for all byte values)
    freq = [0] * 256
    for b in data:
        freq[b] += 1

    encoded = get_arithmetic_offset(data, freq)
    print(f"Encoded offset: {encoded[0]}/{encoded[1]}")

    decoded = get_arithmetic_values(encoded, freq, len(data))
    print(f"Decoded: {decoded}")
    print(f"Match:   {decoded == data}")
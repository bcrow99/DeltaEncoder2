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
# Prime factorization (no external dependencies)
# ---------------------------------------------------------------------------

def get_prime_factors(n: int) -> list[int]:
    """Return prime factors of n as a sorted list with repetition."""
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
        factors.append(n)
    return factors


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

def get_interval_value(
    src: bytes,
    frequency: list[int],
    maximum_range: int = 10_000,
    minimum_range: int = 512,
) -> tuple[int, int]:
    """
    Arithmetic encode src and return the most-reducible fraction within
    the valid encoding interval.

    Rather than returning the raw lower-bound offset, we search the
    interval [offset, offset+range) for the integer offset i that
    maximises gcd(offset_num + i, offset_den).  That candidate reduces
    to the smallest possible (numerator, denominator) pair, meaning
    fewer bits are needed to represent the encoded value.

    Args:
        src:           Bytes to encode.
        frequency:     frequency[i] = count of byte value i in src (256 entries).
        maximum_range: Upper bound on the search-window width after scaling.
        minimum_range: Lower bound on the search-window width after scaling.

    Returns:
        (numerator, denominator) — a reduced fraction inside the encoding
        interval, chosen to be as simple as possible.
    """
    f = frequency[:]
    n = len(src)

    # Build cumulative frequency table
    s = [0] * len(f)
    m = 0
    for i in range(len(f)):
        s[i] = m
        m += f[i]

    # Encode: track interval as two rationals offset/1 and range/1
    off_n, off_d = 0, 1   # offset  = 0/1
    rng_n, rng_d = 1, 1   # range   = 1/1

    for i in range(n):
        j = src[i]

        # addend = range * s[j] / m
        add_n = rng_n * s[j]
        add_d = rng_d * m
        g = gcd(add_n, add_d)
        if g > 1:
            add_n //= g
            add_d //= g

        # offset += addend
        off_n = off_n * add_d + add_n * off_d
        off_d = off_d * add_d
        g = gcd(off_n, off_d)
        if g > 1:
            off_n //= g
            off_d //= g

        # range *= f[j] / m
        rng_n = rng_n * f[j]
        rng_d = rng_d * m
        g = gcd(rng_n, rng_d)
        if g > 1:
            rng_n //= g
            rng_d //= g

        # Adaptive update
        f[j] -= 1
        m -= 1
        for k in range(j + 1, len(s)):
            s[k] -= 1

    # --- Bring offset and range to a common denominator ----------------
    if off_d != rng_d:
        off_n *= rng_d
        rng_n *= off_d
        common_d = off_d * rng_d
        off_d    = common_d
        rng_d    = common_d

    # --- Scale range into [minimum_range, maximum_range] ---------------
    #
    # First try dividing by prime factors shared between offset_num and
    # offset_den (largest first) to shrink the window.
    g            = gcd(off_n, off_d)
    factor_list  = get_prime_factors(g)
    factor       = 1

    j = len(factor_list) - 1
    while rng_n // factor > maximum_range and j >= 0:
        factor *= factor_list[j]
        j -= 1

    if factor > 1:
        off_n //= factor
        off_d //= factor
        rng_n //= factor
        rng_d   = off_d

    # If range is too small, scale up by powers of 2
    if rng_n < minimum_range:
        scale = 2
        while rng_n * scale < minimum_range:
            scale *= 2
        off_n *= scale
        off_d *= scale
        rng_n *= scale
        rng_d  = off_d

    # --- Search for most-reducible fraction in [off_n, off_n + rng_n) -
    best_gcd = gcd(off_n, off_d)
    best_i   = 0

    for i in range(1, rng_n):
        g = gcd(off_n + i, off_d)
        if g > best_gcd:
            best_gcd = g
            best_i   = i

    value_n = (off_n + best_i) // best_gcd
    value_d =  off_d           // best_gcd

    return value_n, value_d


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

def get_arithmetic_values2(v: tuple[int, int], frequency: list[int], n: int) -> bytes:
    """Decode n symbols from rational v using the given frequency table."""
    value = bytearray(n)

    arith_list = []
    m = 0
    for i in range(len(frequency)):
        if frequency[i] != 0:
            arith_list.append([i, frequency[i], m])
            m += frequency[i]

    off_n, off_d = 0, 1
    rng_n, rng_d = 1, 1
    w_n,   w_d   = v

    for i in range(n):
        # w = v - offset
        if off_n != 0:
            w_n = v[0] * off_d - off_n * v[1]
            w_d = v[1] * off_d
            g = gcd(abs(w_n), w_d)
            if g > 1:
                w_n //= g
                w_d //= g

        # Binary search for the symbol whose interval contains w
        j     = len(arith_list) // 2
        entry = arith_list[j]
        f_j   = entry[1]
        s_j   = entry[2]

        d = rng_d * m
        a = rng_n * s_j           * w_d
        b = w_n                   * d
        c = rng_n * (s_j + f_j)  * w_d

        if a > b:
            k = max(j // 2, 1)
            while a > b:
                j -= k
                entry = arith_list[j]
                f_j, s_j = entry[1], entry[2]
                a = rng_n * s_j * w_d
                k = max(k // 2, 1)
            c = rng_n * (s_j + f_j) * w_d
            if c <= b:
                while c <= b:
                    j += 1
                    entry = arith_list[j]
                    f_j, s_j = entry[1], entry[2]
                    c = rng_n * (s_j + f_j) * w_d
        elif c <= b:
            k = max((len(arith_list) - j) // 2, 1)
            while c <= b:
                j += k
                entry = arith_list[j]
                f_j, s_j = entry[1], entry[2]
                c = rng_n * (s_j + f_j) * w_d
                k = max(k // 2, 1)
            a = rng_n * s_j * w_d
            if a > b:
                while a > b:
                    j -= 1
                    entry = arith_list[j]
                    f_j, s_j = entry[1], entry[2]
                    a = rng_n * s_j * w_d

        # Update offset and range
        add_n = rng_n * s_j
        add_d = rng_d * m

        off_n = off_n * add_d + add_n * off_d
        off_d = off_d * add_d
        g = gcd(abs(off_n), off_d)
        if g > 1:
            off_n //= g
            off_d //= g

        rng_n = rng_n * f_j
        rng_d = rng_d * m
        g = gcd(rng_n, rng_d)
        if g > 1:
            rng_n //= g
            rng_d //= g

        for p in range(j + 1, len(arith_list)):
            arith_list[p][2] -= 1

        f_j -= 1
        m   -= 1
        if f_j != 0:
            arith_list[j][1] = f_j
        else:
            arith_list.pop(j)

        value[i] = entry[0]

    return bytes(value)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def build_freq(data: bytes) -> list[int]:
        freq = [0] * 256
        for b in data:
            freq[b] += 1
        return freq

    def test(label: str, data: bytes) -> None:
        if not data:
            return
        freq    = build_freq(data)
        v       = get_interval_value(data, freq)
        decoded = get_arithmetic_values(v, freq, len(data))
        ok      = "✓" if decoded == data else "✗ MISMATCH"
        bits    = max(v[0].bit_length(), v[1].bit_length())
        print(f"{ok}  {label:35s}  encoded={v[0]}/{v[1]}  ({bits} bits)")

    print("=== get_interval_value (standalone) ===\n")
    test("single byte",            b"A")
    test("hello arithmetic world", b"hello arithmetic world")
    test("english pangram",        b"the quick brown fox jumps over the lazy dog")
    test("all same (100×A)",       b"A" * 100)
    test("two symbols (AB×50)",    b"AB" * 50)
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
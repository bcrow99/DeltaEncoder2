import numpy as np

def resize_x(src: np.ndarray, new_xdim: int) -> np.ndarray:
    ydim, xdim = src.shape
    if new_xdim == xdim:
        return src.copy()

    if new_xdim < xdim:
        number_of_segments = xdim - new_xdim + 1
        remainder = new_xdim % number_of_segments
        if remainder == 0:
            segment_length = new_xdim // number_of_segments
            keep_mask = np.ones(xdim, dtype=bool)
            drop = np.arange(segment_length, xdim, segment_length + 1)
            keep_mask[drop] = False
        else:
            number_of_segments = xdim - new_xdim
            remainder = new_xdim % number_of_segments
            segment_length = new_xdim // number_of_segments
            if remainder == 0:
                keep_mask = np.ones(xdim, dtype=bool)
                drop = np.arange(segment_length, xdim, segment_length + 1)
                keep_mask[drop] = False
            else:
                is_long = np.zeros(number_of_segments, dtype=bool)
                interval = 1.0 / (remainder + 1)
                increment = int(interval * number_of_segments)
                indices = np.arange(1, remainder + 1) * increment
                is_long[indices] = True
                seg_lengths = np.where(is_long, segment_length + 1, segment_length)
                drop = np.cumsum(seg_lengths)[:-1]
                keep_mask = np.ones(xdim, dtype=bool)
                keep_mask[drop] = False
        return src[:, keep_mask]

    else:  # new_xdim > xdim
        number_of_segments = new_xdim - xdim + 1
        remainder = xdim % number_of_segments
        if remainder == 0:
            segment_length = xdim // number_of_segments
            insert_after = np.arange(segment_length - 1, xdim - 1, segment_length)
        else:
            number_of_segments = new_xdim - xdim
            remainder = xdim % number_of_segments
            segment_length = xdim // number_of_segments
            if remainder == 0:
                insert_after = np.arange(segment_length - 1, xdim - 1, segment_length)
            else:
                is_long = np.zeros(number_of_segments, dtype=bool)
                interval = 1.0 / (remainder + 1)
                increment = int(interval * number_of_segments)
                indices = np.arange(1, remainder + 1) * increment
                is_long[indices] = True
                seg_lengths = np.where(is_long, segment_length + 1, segment_length)
                insert_after = np.cumsum(seg_lengths)[:-1] - 1

        repeats = np.ones(xdim, dtype=int)
        repeats[insert_after] = 2
        dst = src.repeat(repeats, axis=1)

        offsets = np.arange(len(insert_after))
        interp_cols = insert_after + offsets + 1
        dst[:, interp_cols] = ((src[:, insert_after].astype(np.float32) +
                                src[:, insert_after + 1].astype(np.float32)) / 2
                               ).astype(np.uint8)
        return dst


def resize_y(src: np.ndarray, new_ydim: int) -> np.ndarray:
    ydim, xdim = src.shape
    if new_ydim == ydim:
        return src.copy()

    if new_ydim < ydim:
        number_of_segments = ydim - new_ydim + 1
        remainder = new_ydim % number_of_segments
        if remainder == 0:
            segment_length = new_ydim // number_of_segments
            keep_mask = np.ones(ydim, dtype=bool)
            drop = np.arange(segment_length, ydim, segment_length + 1)
            keep_mask[drop] = False
        else:
            number_of_segments = ydim - new_ydim
            remainder = new_ydim % number_of_segments
            segment_length = new_ydim // number_of_segments
            if remainder == 0:
                keep_mask = np.ones(ydim, dtype=bool)
                drop = np.arange(segment_length, ydim, segment_length + 1)
                keep_mask[drop] = False
            else:
                is_long = np.zeros(number_of_segments, dtype=bool)
                interval = 1.0 / (remainder + 1)
                increment = int(interval * number_of_segments)
                indices = np.arange(1, remainder + 1) * increment
                is_long[indices] = True
                seg_lengths = np.where(is_long, segment_length + 1, segment_length)
                drop = np.cumsum(seg_lengths)[:-1]
                keep_mask = np.ones(ydim, dtype=bool)
                keep_mask[drop] = False
        return src[keep_mask, :]

    else:  # new_ydim > ydim
        number_of_segments = new_ydim - ydim + 1
        remainder = ydim % number_of_segments
        if remainder == 0:
            segment_length = ydim // number_of_segments
            insert_after = np.arange(segment_length - 1, ydim - 1, segment_length)
        else:
            number_of_segments = new_ydim - ydim
            remainder = ydim % number_of_segments
            segment_length = ydim // number_of_segments
            if remainder == 0:
                insert_after = np.arange(segment_length - 1, ydim - 1, segment_length)
            else:
                is_long = np.zeros(number_of_segments, dtype=bool)
                interval = 1.0 / (remainder + 1)
                increment = int(interval * number_of_segments)
                indices = np.arange(1, remainder + 1) * increment
                is_long[indices] = True
                seg_lengths = np.where(is_long, segment_length + 1, segment_length)
                insert_after = np.cumsum(seg_lengths)[:-1] - 1

        repeats = np.ones(ydim, dtype=int)
        repeats[insert_after] = 2
        dst = src.repeat(repeats, axis=0)

        offsets = np.arange(len(insert_after))
        interp_rows = insert_after + offsets + 1
        dst[interp_rows, :] = ((src[insert_after, :].astype(np.float32) +
                                src[insert_after + 1, :].astype(np.float32)) / 2
                               ).astype(np.uint8)
        return dst

def resize(src: np.ndarray, new_xdim: int, new_ydim: int) -> np.ndarray:
    dim = src.shape
    ydim = dim[0]
    xdim = dim[1]
    
    if new_xdim * new_ydim < xdim * ydim:
        tmp = resize_x(src, new_xdim)
        return resize_y(tmp, new_ydim)
    else:
        tmp = resize_y(src, new_ydim)
        return resize_x(tmp, new_xdim)
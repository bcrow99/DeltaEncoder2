import sys
import cv2
import time
import numpy as np
import delta
import resize
import string2
import zlib


file = sys.argv[1]
print(file)

# Load the image
image = cv2.imread(file)

ydim, xdim, number_of_channels = image.shape

pixel_quant = 3
factor: float = pixel_quant
factor /= 10
new_xdim = xdim - (int) (factor * (xdim / 2 - 2))
new_ydim = ydim - (int) (factor * (ydim / 2 - 2))

pixel_shift = 3

total_compressed_bytes = 0

start        = time.perf_counter_ns()
channel_list = list()
for i in range(3):
    channel = image[:, :, i]
    shape   = channel.shape
    shrunken_channel          = resize.resize(channel, new_xdim, new_ydim)
    shifted_channel           = delta.shift(shrunken_channel, -pixel_shift)
    init_value, shifted_delta = delta.get_horizontal_deltas_from_values(shifted_channel)
    
    min_value = shifted_delta.min() 
    shifted_delta -= min_value
    shifted_delta[0,0] = 0;
    
    histogram = string2.get_histogram(shifted_delta)
    rank_table = string2.get_rank_table(histogram)
    bitlength, packed_delta = string2.pack_strings(shifted_delta, rank_table)
    
    compressed_packed_delta = zlib.compress(packed_delta)
    total_compressed_bytes += len(compressed_packed_delta)
    decompressed_packed_delta = zlib.decompress(compressed_packed_delta)
    
    number_unpacked, unpacked_delta = string2.unpack_strings(decompressed_packed_delta, rank_table, shape, bitlength)
    unpacked_delta       += min_value
    unpacked_delta[0, 0]  = 0
    reconstructed_channel = delta.get_values_from_horizontal_deltas(unpacked_delta, init_value)
    reshifted_channel     = delta.shift(shifted_channel, pixel_shift)
    expanded_channel      = resize.resize(reshifted_channel, xdim, ydim)
    
    channel_list.append(expanded_channel)
resized_image = cv2.merge((channel_list[0], channel_list[1], channel_list[2]))   
stop          = time.perf_counter_ns()

elapsed_time = stop - start
elapsed_time /= 1000000;

compression_rate = float(total_compressed_bytes)
compression_rate /= (xdim * ydim * 3)

print("Compression rate is ", compression_rate)
print("It took ",  elapsed_time, " ms to process image.");

cv2.imshow("Writer", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()  


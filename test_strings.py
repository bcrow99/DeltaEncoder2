import sys
import cv2
import time
import numpy as np
import delta
import resize
import string2


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

start        = time.perf_counter_ns()
channel_list = list()
for i in range(3):
    channel = image[:, :, i]
    shape = channel.shape
    init_value, pixel_delta = delta.get_horizontal_deltas_from_values(channel)
    
    min_value = pixel_delta.min()
    pixel_delta -= min_value
    pixel_delta[0,0] = 0;
    
    histogram = string2.get_histogram(pixel_delta)
    rank_table = string2.get_rank_table(histogram)
    
    bitlength, packed_delta = string2.pack_strings(pixel_delta, rank_table)
    number_unpacked, unpacked_delta = string2.unpack_strings(packed_delta, rank_table, shape, bitlength)
    unpacked_delta += min_value
    unpacked_delta[0,0] = 0
    reconstructed_channel = delta.get_values_from_horizontal_deltas(unpacked_delta, init_value)
    
    channel_list.append(reconstructed_channel)
resized_image = cv2.merge((channel_list[0], channel_list[1], channel_list[2]))   
stop          = time.perf_counter_ns()

elapsed_time = stop - start
elapsed_time /= 1000000;
print("It took ",  elapsed_time, " ms to process image.");

cv2.imshow("Test Strings", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()  


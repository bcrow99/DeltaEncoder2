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


pixel_quant = 2
factor: float = pixel_quant
factor /= 10
new_xdim = xdim - (int) (factor * (xdim / 2 - 2))
new_ydim = ydim - (int) (factor * (ydim / 2 - 2))

pixel_shift = 3


channel_list = list()
for i in range(3):
    channel = image[:, :, i]
    shrunken_channel  = resize.resize(channel, new_xdim, new_ydim)
    shifted_channel   = delta.shift(shrunken_channel, -pixel_shift)
    
    reshifted_channel = delta.shift(shifted_channel, pixel_shift)
    expanded_channel = resize.resize(reshifted_channel, xdim, ydim)
    channel_list.append(expanded_channel)
resized_image = cv2.merge((channel_list[0], channel_list[1], channel_list[2]))   

cv2.imshow("Writer", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()  


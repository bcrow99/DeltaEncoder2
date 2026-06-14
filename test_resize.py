import sys
import time
import array
import cv2
import numpy as np
import math
import delta.resize as resize

xdim = 512
ydim = 512
gray = np.zeros((ydim, xdim), dtype=np.uint8) 

for i in range(0, ydim):
    for j in range(0, xdim):
        gray[i, j] = j // 2       

new_xdim = 367
new_ydim = 367


start = time.perf_counter_ns()
shrunken_gray = resize.resize(gray, new_xdim, new_ydim)
resized_gray  = resize.resize(shrunken_gray, xdim, ydim)
stop = time.perf_counter_ns()

elapsed_time = stop - start
elapsed_time /= 2
elapsed_time /= 1000;

elapsed_time = math.floor(elapsed_time)

print("It took ",  elapsed_time, " usec to resize grayscale");


cv2.imshow("Grayscale", resized_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()    
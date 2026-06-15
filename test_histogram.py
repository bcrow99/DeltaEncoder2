import sys
import time
import math
import cv2
import numpy as np
import delta
import string2

gray = np.zeros((512, 1024), dtype=np.uint8) 

for i in range(0, 512):
    for j in range(0, 1024):
        gray[i, j] = j / 4
shift = -3;
shifted_gray   = delta.shift(gray, shift);
quantized_gray = delta.shift(shifted_gray, -shift);

start = time.perf_counter_ns()
for i in range(100):
    histogram = string2.get_histogram(shifted_gray)
stop = time.perf_counter_ns()

i = 0;
while(histogram[i] == 0):
    i += 1;
j = i
print("Min value is ", j)

i = histogram.size - 1;
while(histogram[i] == 0):
    i -= 1;
k = i
print("Max value is ", k)

i = k - j
print("Range is ", i)

print("Completed histogram test.")

elapsed_time = stop - start
elapsed_time /= 100
elapsed_time /= 1000000;
elapsed_time = math.floor(elapsed_time)
print(elapsed_time, " ms to get histogram.")  

cv2.imshow("Grayscale", quantized_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()   
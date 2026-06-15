'''
Created on Jun 9, 2026

@author: bcrow
'''
import sys
import time
import math
import cv2
import numpy as np
import delta

'''
file = sys.argv[1]

img = cv2.imread(file)

ydim, xdim, number_of_channels = img.shape

b,g,r = cv2.split(img)

pixel_shift = 3;
'''
gray = np.zeros((512, 1024), dtype=np.uint8) 

for i in range(0, 512):
    for j in range(0, 1024):
        gray[i, j] = j / 4
        
start = time.perf_counter_ns()

shift = -3;
shifted_gray   = delta.shift(gray, shift);
quantized_gray = delta.shift(shifted_gray, -shift);

for i in range(0, 99):
    shifted_gray   = delta.shift(gray, shift);
    quantized_gray = delta.shift(shifted_gray, -shift);
    
stop = time.perf_counter_ns()

elapsed_time = stop - start
elapsed_time /= 200
elapsed_time /= 1000;
elapsed_time = math.floor(elapsed_time)
print(elapsed_time, "usecs to shift gray scale: ")  

cv2.imshow("Grayscale", quantized_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()     
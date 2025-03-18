from pickletools import uint8

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.core.pylabtools import figsize

img_bgr = cv2.imread("New_Zealand_Coast.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.show()

#Addition or Brightness
matrix = np.ones(img_rgb.shape, dtype="uint8") * 50

img_rgb_brighter = cv2.add(img_rgb, matrix)
img_rgb_darker = cv2.subtract(img_rgb, matrix)

plt.figure(figsize=[18, 5])
plt.subplot(131);plt.imshow(img_rgb_brighter);plt.title("Brighter")
plt.subplot(132);plt.imshow(img_rgb_darker);plt.title("Darker")
plt.subplot(133);plt.imshow(img_rgb);plt.title("Original")
plt.show()

#Multiplication or Contrast
matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_brighter = np.uint8(cv2.multiply(np.float64(img_rgb), matrix2))

plt.figure(figsize=[18, 5])
plt.subplot(131);plt.imshow(img_rgb_darker);plt.title("Lower Contrast")
plt.subplot(132);plt.imshow(img_rgb);plt.title("Original")
plt.subplot(133);plt.imshow(img_rgb_brighter);plt.title("Higher Contrast")
plt.show()

#Handling Overflow using np.clip
matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_lower  = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))

plt.figure(figsize=[18, 5])
plt.subplot(131);plt.imshow(img_rgb_lower);plt.title("Lower Contrast")
plt.subplot(132);plt.imshow(img_rgb);plt.title("Original")
plt.subplot(133);plt.imshow(img_rgb_higher);plt.title("Higher Contrast")
plt.show()

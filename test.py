import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

img_NZ_bgr = cv2.imread("New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)
b, g, r = cv2.split(img_NZ_bgr)

plt.figure(figsize=[20,5])

plt.subplot(141);plt.imshow(r, cmap="gray");plt.title("Red Channel")
plt.subplot(142);plt.imshow(g, cmap="gray");plt.title("Green Channel")
plt.subplot(143);plt.imshow(b, cmap="gray");plt.title("Blue Channel")

imgMerged = cv2.merge((b, g, r))

plt.subplot(144)
plt.imshow(imgMerged[:,:,::-1])
plt.title("Merged Output")
plt.show()
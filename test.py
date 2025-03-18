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

#Changing from BGR to RGB
img_NZ_rgb = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_NZ_rgb)
plt.show()

#Changing to HSV Color Space
img_hsv = cv2.cvtColor(img_NZ_bgr, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(img_hsv)

plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel");
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel");
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel");
plt.subplot(144);plt.imshow(img_NZ_rgb); plt.title("Original")
plt.show()

#Modifying individual channel

h_new = h + 10
imgMerged = cv2.merge((h_new, s, v))
img_NZ_rgb = cv2.cvtColor(imgMerged, cv2.COLOR_HSV2RGB)

plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h, cmap="gray");plt.title("H Channel");
plt.subplot(142);plt.imshow(s, cmap="gray");plt.title("S Channel");
plt.subplot(143);plt.imshow(v, cmap="gray");plt.title("V Channel");
plt.subplot(144);plt.imshow(img_NZ_rgb);plt.title("Original")
plt.show()

cv2.imwrite("New_Zealand_Lake_SAVED.png", img_NZ_bgr)
image(filename='New_Zealand_Lake_SAVED.png')



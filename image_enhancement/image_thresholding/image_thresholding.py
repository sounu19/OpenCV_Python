import cv2
import matplotlib.pyplot as plt

from image_enhancement.basic_image_enhancement import img_bgr

img_read = cv2.imread("building-windows.jpg", cv2.IMREAD_GRAYSCALE)
retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)

plt.figure(figsize=[18, 5])
plt.subplot(121),plt.imshow(img_read, cmap="gray");plt.title("Original")
plt.subplot(122),plt.imshow(img_thresh, cmap="gray");plt.title("Thresholded")
print(img_thresh.shape)
plt.show()

#Application: Sheet Music Reader
img_read = cv2.imread("Piano_Sheet_Music.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("test",img_read)
cv2.waitKey(0)

retval, img_thresh_gbl_1 = cv2.threshold(img_read, 50, 255 , cv2.THRESH_BINARY)

retval, img_thresh_gbl_2 = cv2.threshold(img_read, 130, 255 , cv2.THRESH_BINARY)

img_thresh_adp = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

plt.figure(figsize=[18, 15])
plt.subplot(221);plt.imshow(img_read, cmap="gray");plt.title("Original")
plt.subplot(222);plt.imshow(img_thresh_gbl_1, cmap="gray");plt.title("Thresholded (global: 50)")
plt.subplot(223);plt.imshow(img_thresh_gbl_2, cmap="gray");plt.title("Thresholded (global: 130)")
plt.subplot(224);plt.imshow(img_thresh_adp, cmap="gray");plt.title("Thresholded (Adaptive)")
plt.show()



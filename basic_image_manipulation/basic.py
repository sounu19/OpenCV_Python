import cv2
import matplotlib.pyplot as plt

cb_img = cv2.imread("checkerboard_18x18.png")

plt.imshow(cb_img, cmap="gray")
print(cb_img)
print(cb_img[0, 0])

cb_img_copy = cb_img.copy()
cb_img_copy[2, 2] = 200
cb_img_copy[2, 3] = 200
cb_img_copy[3, 2] = 200
cb_img_copy[3, 3] = 200

plt.imshow(cb_img_copy, cmap="gray")
print(cb_img_copy)
plt.show()




import cv2
import matplotlib.pyplot as plt

img_rec = cv2.imread("rectangle.jpg", cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread("circle.jpg", cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=[20, 5])
plt.subplot(121);plt.imshow(img_rec, cmap="gray")
plt.subplot(122);plt.imshow(img_cir, cmap="gray")
print(img_rec.shape)
print(img_cir.shape)
plt.show()

#Bitwise AND Operator
result = cv2.bitwise_and(img_cir, img_rec, mask=None)
plt.imshow(result, cmap="gray")
plt.show()

#Bitwise OR Operator
result = cv2.bitwise_or(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray")
plt.show()

#Bitwise XOR Operator
result = cv2.bitwise_xor(img_rec, img_cir, mask=None)
plt.imshow(result, cmap="gray")
plt.show()
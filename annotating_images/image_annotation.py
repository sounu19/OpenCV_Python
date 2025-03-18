import cv2
import matplotlib.pyplot as plt

image = cv2.imread("Apollo_11_Launch.jpg", cv2.IMREAD_COLOR)
plt.imshow(image[:, :, ::-1])
plt.show()

#Drawing a Line
image_line = image.copy()

cv2.line(image_line, (300,200), (300,400), (0,0,255), thickness=5, lineType=cv2.LINE_AA)
cv2.putText(image_line, "Drawing Line for fun", (200, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
plt.imshow(image_line[:, :, ::-1])
plt.show()

#Drawing a Circle
image_circle = image.copy()

cv2.circle(image_circle,(478,266),250,(0,0,255), thickness=5, lineType=cv2.LINE_AA)
plt.imshow(image_circle[:, :, ::-1])
plt.show()

#Drawing a Rectangle

image_rectangle = image.copy()

cv2.rectangle(image_rectangle, (400, 50), (600, 500), (0, 0, 255), thickness=-5, lineType=cv2.LINE_AA)
plt.imshow(image_rectangle[:, :, ::-1])
plt.show()

#Adding Text
image_text = image.copy()
text = "Apollo 11 Saturn V Launch, July 16, 1969"
fontScale = 1.2
fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
fontColor = (0, 255, 0)
fontThickness = 2

cv2.putText(image_text, text, (200, 550), fontFace, fontScale, fontColor, fontThickness, lineType=cv2.LINE_AA);
plt.imshow(image_text[:, :, ::-1])
plt.show()

cv2.LINE
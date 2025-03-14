import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import Image

coke_img = cv2.imread("coca-cola-logo.png", 1)
'''print("Image size (H, W, C) is: ", coke_img.shape)
print("Image data type is: ", coke_img.dtype)'''
'''coke_img_rgb = cv2.cvtColor(coke_img, cv2.COLOR_BGR2RGB)'''
plt.imshow(coke_img)
plt.show()
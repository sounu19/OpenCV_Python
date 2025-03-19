import cv2
import matplotlib.pyplot as plt
from IPython.core.display_functions import display
from IPython.lib.display import YouTubeVideo
import ffmpeg

source = 'race_car.mp4'

cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("Error Opening Video Stream or File")

ret, frame = cap.read()
plt.imshow(frame[..., ::-1])
plt.show()

#Display the video file
video = YouTubeVideo("RwxVEjv78LQ", width=700, height=438)
display(video)

#Write Video using OpenCV
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out_avi = cv2.VideoWriter("race_car_out.avi", cv2.VideoWriter_fourcc("M", "J", "P", "G"), 10, (frame_width, frame_height))
out_mp4 = cv2.VideoWriter("race_car_out.mp4", cv2.VideoWriter_fourcc(*"XVID"), 10, (frame_width, frame_height))
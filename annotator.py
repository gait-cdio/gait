import numpy as np
import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('4farger.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('4farger.avi', fourcc, fps, (width, height))
counter = 0

cv2.namedWindow('Keypoints',cv2.WINDOW_NORMAL)
cv2.resizeWindow('Keypoints', 800,600)

paused = True

while (cap.isOpened()):

    ret, img = cap.read()

    if ret == True:

        cv2.imshow("Image", img)

        if paused:
            delay = 0
        else:
            delay = 1

        pressed_key = cv2.waitKey(delay) & 0xFF
        if pressed_key == ord(' '):
            paused = not paused
        elif pressed_key == ord('q'):
            break
    else:
        break
    counter += 1

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
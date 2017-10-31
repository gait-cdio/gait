import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

from collections import namedtuple

from colortracker import match_points, PointFeatures

plt.ioff()
# import PyOpenPose

videoname = '4farger'
cache_filename = videoname + '.detections.npy'

tracker = namedtuple('tracker', ['tracker', 'lower_bound', 'upper_bound', 'name'])

if 'cached' in sys.argv and os.path.isfile(cache_filename):
    pointbuffer = np.load(cache_filename)
else:
    cap = cv2.VideoCapture(videoname + '.mp4')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(videoname +'.avi', fourcc, fps, (width, height))
    counter = 0
    pointbuffer = []

    ret,img=cap.read()

    cv2.namedWindow('Keypoints',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Keypoints', 800,600)

    paused = True

    t_coords = []
    x_coords = []
    y_coords = []

    backgroundModel = cv2.createBackgroundSubtractorMOG2()

    while (cap.isOpened()):

        ret, img = cap.read()
        if ret == True:

            mask = backgroundModel.apply(img)

            cv2.imshow('Keypoints',mask)

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
    counter = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while (cap.isOpened()):

        ret, img = cap.read()
        if ret == True:

            mask = backgroundModel.apply(img)

            cv2.imshow('Keypoints',mask)

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



def extract_position_square(item):
    return item.position[0] ** 2 + item.position[1] ** 2


# position_variances = [np.var(list(map(extract_position_square, s))) for s in pointString]
# plt.plot(np.log(position_variances), 'o')
# plt.show()

#pointString = [s for s in pointString if len(s) > 10]

f, axes = plt.subplots(ncols=2)

for index in range(0, len(pointbuffer)):
    curve = pointbuffer[index]
    t_c=curve[2]
    x_c=curve[0]
    y_c=curve[1]
    xline = axes[0].plot(t_c, x_c, 'o-', markersize=2)
    yline = axes[1].plot(t_c, y_c, 'o-', markersize=2)

axes[1].invert_yaxis()
plt.show()


print('You did the thing :)')


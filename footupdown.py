import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

from collections import namedtuple

from colortracker import match_points, PointFeatures

plt.ioff()
# import PyOpenPose

ColorInterval = namedtuple('ColorInterval', ['mean_hsv', 'variance'])

videoname = '4farger'

vidpoints = videoname + '.detections.npy'
vidpointsmatched = videoname + '.detections.matched.npy'

points = np.load(vidpoints)
pointsMatched = np.load(vidpointsmatched)

f, axes = plt.subplots(ncols=2)

for index in range(0, len(pointsMatched)):
    curve = pointsMatched[index]
    t_c=[p.frame for p in curve]
    x_c=[p.position[0] for p in curve]
    y_c=[p.position[1] for p in curve]
    xline = axes[0].plot(t_c, x_c, 'o-', markersize=2)
    yline = axes[1].plot(t_c, y_c, 'o-', markersize=2)

axes[1].invert_yaxis()
plt.show()
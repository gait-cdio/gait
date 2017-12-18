import pickle
import itertools
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import cv2
import colortracker
import tracker
import numpy as np
import scipy.ndimage.filters as filt
from scipy import signal

cap = cv2.VideoCapture('input-videos/4farger.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
writer = cv2.VideoWriter('output-videos/only_yellow_occlusions.avi', fourcc, fps, (width, height))

with open('TrackerResults/4farger.mp4.detections.pkl', 'rb') as f:
    detections = pickle.load(f)

ts = list(range(len(detections[0])))
frames = [list(itertools.chain(*[d[frame] for d in detections])) for frame in ts]

positions = [[point.position for point in frame_entries] for frame_entries in frames]
colors = [[matplotlib.colors.hsv_to_rgb((point.hue/180, 1, 0.9)) for point in frame_entries] for frame_entries in frames]

tracks = []
for detection_tracker in detections:
    tracks += tracker.points_to_tracks(detection_tracker,
                                       dist_fun=colortracker.feature_distance(hue_weight=2,
                                                                              size_weight=2,
                                                                              time_weight=1),
                                       similarity_threshold=140)

points = []
for track in tracks:
    points.append({
        't': [state.frame for state in track.state_history],
        'x': [state.x for state in track.state_history],
        'y': [state.y for state in track.state_history],
        'obs': [state.observed for state in track.state_history],
        'color': matplotlib.colors.hsv_to_rgb((track.feature.hue/180, 1, 0.9)),
    })

number_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for frame_nr in range(number_frames):
    ret, image = cap.read()

    for point_index, point_track in enumerate(points):
        if point_index == 2:
            last_coords = None
            for index, frame in enumerate(range(point_track['t'][0], min(frame_nr, point_track['t'][-1]) + 1)):
                assert frame == point_track['t'][index]
                coords = int(point_track['x'][index]), int(point_track['y'][index])
                color = tuple(reversed(point_track['color'] * 255 * (0.5 + 0.5 * int(point_track['obs'][index]))))
                cv2.circle(image, coords, 3, color, thickness=-1)
                if last_coords:
                    cv2.line(image, last_coords, coords, color, thickness=2)
                last_coords = coords

    writer.write(image)
    cv2.imshow('stuff', image)
    cv2.waitKey(0)

cap.release()
writer.release()

import pickle
import itertools
import matplotlib
import matplotlib.pyplot as plt
import cv2
import colortracker
import tracker
import numpy as np
import scipy.ndimage.filters as filt
from scipy import signal
from footupdown import inpaint_1d

cap = cv2.VideoCapture('input-videos/4fargersilly.mp4')
end_frame = 140
cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame)

ret, image = cap.read()
assert ret

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

with open('TrackerResults/4fargersilly.mp4.detections.pkl', 'rb') as f:
    detections = pickle.load(f)

ts = list(range(len(detections[0])))
frames = [list(itertools.chain(*[d[frame] for d in detections])) for frame in ts]

positions = [[point.position for point in frame_entries] for frame_entries in frames]
colors = [[matplotlib.colors.hsv_to_rgb((point.hue/180, 1, 0.9)) for point in frame_entries] for frame_entries in frames]

fig1 = plt.figure()

plt.imshow(image)

xs = [p[0] for ps in positions[:end_frame + 1] for p in ps]
ys = [p[1] for ps in positions[:end_frame + 1] for p in ps]
cs = [color for cs in colors for color in cs]
plt.scatter(xs, ys, s=5, c=cs)

plt.show()

fig2 = plt.figure()
axis = fig2.subplots(nrows=1, ncols=2, sharex=True)

for t in ts:
    point_colors = colors[t]

    axis[0].scatter([t] * len(point_colors), [p[0] for p in positions[t]], c=point_colors, s=5)
    axis[1].scatter([t] * len(point_colors), [p[1] for p in positions[t]], c=point_colors, s=5)

axis[0].set_xlim(157, 263)
axis[0].set_ylim(0, 650)
axis[1].set_ylim(725, 512)

axis[0].set_facecolor('xkcd:dark grey')
axis[1].set_facecolor('xkcd:dark grey')

axis[0].set_xlabel('frame')
axis[0].set_ylabel('x coordinate')
axis[1].set_xlabel('frame')
axis[1].set_ylabel('y coordinate')

plt.show() # Plot the detections for presentation

tracks = []
for detection_tracker in detections:
    tracks += tracker.points_to_tracks(detection_tracker,
                                       dist_fun=colortracker.feature_distance(hue_weight=2,
                                                                              size_weight=2,
                                                                              time_weight=1),
                                       similarity_threshold=140)


f, axes = plt.subplots(ncols=2, nrows=1, sharex=True, sharey=True)
for track_index in range(0, 4):
    track_index = int(track_index)
    point_track = tracks[track_index]

    t = [state.frame for state in point_track.state_history]
    x = [state.x for state in point_track.state_history]

    xline = axes[0].plot(t, x, 'o-', markersize=2, label='x position, index ' + str(track_index), c=colors[50][track_index])

for t in ts:
    point_colors = colors[t]
    axes[1].scatter([t] * len(point_colors), [p[0] for p in positions[t]], c=point_colors, s=5)


axes[0].set_facecolor('xkcd:dark grey')
axes[1].set_facecolor('xkcd:dark grey')

axes[0].set_xlim(115, 230)
axes[0].set_ylim(350, 1150)

axes[0].set_xlabel('frame')
axes[0].set_ylabel('x coordinate, tracks')
axes[1].set_xlabel('frame')
axes[1].set_ylabel('x coordinate, detections')

plt.show()

f, axes = plt.subplots(ncols=2, nrows=1, sharex=True)

for track_index, track in enumerate(tracks):
    x_vals = np.asarray([state.x for state in track.state_history])
    t_vals = np.asarray([state.frame for state in track.state_history])


    #x_c = signal.detrend([state.x for state in track.state_history])
    #t_c = [state.frame for state in track.state_history]

    t_first = t_vals[0]
    x_c = signal.detrend([x_vals[max(80-t_first, 0):]])
    t_c = t_vals[max(80-t_first,0):]


    y_c = signal.detrend([state.y for state in track.state_history])
    dx_c = filt.gaussian_filter1d(input=x_c, sigma=3, order=1,
                                  mode='nearest')  # order=1 lagpass + derivering. TODO explore mode options

    xline = axes[0].plot(t_c, np.squeeze(x_c), 'o-', markersize=2, label='detrended x position, index ' + str(track_index),
                         c=colors[50][track_index])
    xline = axes[1].plot(t_c, np.squeeze(dx_c), 'o-', markersize=2, label='detrended x position, index ' + str(track_index),
                         c=colors[50][track_index])

axes[0].set_facecolor('xkcd:dark grey')
axes[1].set_facecolor('xkcd:dark grey')
axes[0].set_xlabel('frame')
axes[0].set_ylabel('detrended x coordinate')
axes[1].set_xlabel('frame')
axes[1].set_ylabel('derivative of detrended x coordinate')
axes[0].grid(color='xkcd:grey', linestyle='-')
axes[1].grid(color='xkcd:grey', linestyle='-')
axes[0].set_xlim(115, 235)

plt.show()





import cv2
import numpy as np

from colortracker import ColorTracker
from footupdown import estimate_naive
import matplotlib.pyplot as plt
import utils
import os.path

plt.ioff()

# Load videostream
# video_stream = load_video() | stream_from_webcam()

filename = '4farger.mp4'  # TODO: parse arguments for this
cap = cv2.VideoCapture(filename)

# Initialize stuff
# Select marker/-less
# Maybe prompt user to select colors, regions, etc.
# Create instances of necessary classes (SimpleBlobDetector, TrackerKCF, etc.)
keypoint_tracker = ColorTracker()

paused = False

# Detect keypoints (heel, toe) in each frame
detections = []
frame_nr = 0
while cap.isOpened():
    ret, img = cap.read()
    if ret:
        detections_for_frame = keypoint_tracker.detect(img, frame_nr)
        detections.append(detections_for_frame)

        if paused:
            delay = 0
        else:
            delay = 1

        pressed_key = cv2.waitKey(delay) & 0xFF
        if pressed_key == ord(' '):
            paused = not paused
        elif pressed_key == ord('q'):
            break
        frame_nr += 1
    else:
        keypoint_tracker.cleanup_windows()
        break

# Associate keypoints to form tracks
tracks = keypoint_tracker.associate(detections)
np.save(filename + '_quite_nice', tracks)

# TODO(rolf): make this plotting code prettier

fig, axes = plt.subplots(ncols=2)

for index in range(0, len(tracks)):
    curve = tracks[index]
    t_c = [p.frame for p in curve]
    x_c = [p.position[0] for p in curve]
    y_c = [p.position[1] for p in curve]
    axes[0].plot(t_c, x_c, 'o-', markersize=2)
    axes[1].plot(t_c, y_c, 'o-', markersize=2)

axes[1].invert_yaxis()
plt.show()

# TODO(rolf): link the subplots in some way to easily see which points correspond,
# for example by highlighting the same x value in both subplots when hovering a point in one subplot

# Generate foot down/up

updown_estimations = estimate_naive(tracks)

# Present results


f, axes = plt.subplots(ncols=2)

for track_index in range(0, len(updown_estimations)):
    updown_estimation = updown_estimations[track_index]
    point_track = tracks[track_index]
    estdxline = axes[0].plot((1 + index) * 1000 * updown_estimation, 'o-', markersize=2,
                             label='estimated up/down, index ' + str(track_index))
    estdyline = axes[1].plot(750 - (1 + index) * 100 * updown_estimation, 'o-', markersize=2,
                             label='estimated up/down, index ' + str(track_index))

    t = [p.frame for p in point_track]
    x = [p.position[0] for p in point_track]

    xline = axes[0].plot(t, x, 'o-', markersize=2, label='x position, index ' + str(track_index))

filename_base = os.path.splitext(filename)[0]
groundtruth_filename = filename_base + '.npy'
if os.path.isfile(groundtruth_filename):
    footstates = np.load(groundtruth_filename)
    updown_groundtruth = utils.annotationToOneHot(footstates)

    xleftfoot = axes[0].plot(3000 * updown_groundtruth[0, :], 'o-', markersize=2, label='ground truth up/down')
    yleftfoot = axes[1].plot(750 - 300 * updown_groundtruth[0, :], 'o-', markersize=2, label='ground truth up/down')
else:
    print('WARNING: could not find ground truth for foot up/down')

axes[0].legend()
axes[1].legend()

axes[0].grid(linestyle='-')
axes[1].grid(linestyle='-')

axes[1].invert_yaxis()
plt.show()

print('Done and done.')

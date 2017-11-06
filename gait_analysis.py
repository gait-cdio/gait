import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import sys

import tracker
import utils
import os.path
from gui import set_threshold

from colortracker import ColorTracker
import colortracker
from footupdown import estimate_naive
from tracker import Track, match

plt.ioff()

# Load videostream
# video_stream = load_video() | stream_from_webcam()

filename = '4farger.mp4'  # TODO: parse arguments for this
cache_filename = filename + '.detections.npy'

if 'cached' in sys.argv and os.path.isfile(cache_filename):
    detections = np.load(cache_filename)
    number_frames = len(detections)
else:
    video_reader = imageio.get_reader(filename)
    number_frames = video_reader.get_meta_data()['nframes']

    # Initialize stuff
    # Select marker/-less
    # Maybe prompt user to select colors, regions, etc.
    # Create instances of necessary classes (SimpleBlobDetector, TrackerKCF, etc.)

    keypoint_tracker = ColorTracker()
    keypoint_tracker.hsv_min, keypoint_tracker.hsv_max = set_threshold(video_reader)

    font = cv2.FONT_HERSHEY_TRIPLEX

    paused = False

    # Detect keypoints (heel, toe) in each frame
    detections = []
    frame_nr = 0
    try:
        for frame_nr, img_rgb in enumerate(video_reader):
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(img, str(frame_nr), (10, 30), fontFace=font, fontScale=1, color=(0, 0, 255))
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
    except(RuntimeError):
        print("Video has finished?")
    keypoint_tracker.cleanup_windows()

    # Associate keypoints to form tracks
tracks = tracker.points_to_tracks(detections, dist_fun=colortracker.feature_distance, similarity_threshold=100)
np.save(cache_filename, detections)

# TODO(rolf): make this plotting code prettier

fig, axes = plt.subplots(ncols=2)

for track in tracks:
    t_c = [state.frame for state in track.state_history]
    x_c = [state.x for state in track.state_history]
    y_c = [state.y for state in track.state_history]
    axes[0].plot(t_c, x_c, 'o-', markersize=2)
    axes[1].plot(t_c, y_c, 'o-', markersize=2)

axes[1].invert_yaxis()
plt.show()

# TODO(rolf): link the subplots in some way to easily see which points correspond,
# for example by highlighting the same x value in both subplots when hovering a point in one subplot

# Generate foot down/up

updown_estimations, x_derivatives = estimate_naive(tracks, max_frame=number_frames)

# Present results


f, axes = plt.subplots(ncols=2, nrows=2, sharex=True)

for track_index, point_track in enumerate(tracks):
    updown_estimation = updown_estimations[track_index]
    estdxline = axes[0, 0].plot((1 + track_index) * 1000 * updown_estimation, 'o-', markersize=2,
                             label='estimated up/down, index ' + str(track_index))
    estdyline = axes[0, 1].plot(750 - (1 + track_index) * 100 * updown_estimation, 'o-', markersize=2,
                             label='estimated up/down, index ' + str(track_index))
    derivline = axes[1, 0].plot(range(0, number_frames), x_derivatives[track_index], 'o-', markersize=2)
    t = [state.frame for state in point_track.state_history]
    x = [state.x for state in point_track.state_history]

    xline = axes[0, 0].plot(t, x, 'o-', markersize=2, label='x position, index ' + str(track_index))

filename_base = os.path.splitext(filename)[0]
groundtruth_filename = filename_base + '.npy'
if os.path.isfile(groundtruth_filename):
    footstates = np.load(groundtruth_filename)
    updown_groundtruth = utils.annotationToOneHot(footstates)

    xleftfoot = axes[0, 0].plot(3000 * updown_groundtruth[0, :], 'o-', markersize=2, label='ground truth up/down, left foot')
    yleftfoot = axes[0, 1].plot(750 - 300 * updown_groundtruth[0, :], 'o-', markersize=2, label='ground truth up/down, left foot')
    xrightfoot = axes[0, 0].plot(3000 * updown_groundtruth[1, :], 'o-', markersize=2, label='ground truth up/down, right foot')
    yrightfoot = axes[0, 1].plot(750 - 300 * updown_groundtruth[1, :], 'o-', markersize=2, label='ground truth up/down, right foot')
else:
    print('WARNING: could not find ground truth for foot up/down')

axes[0, 0].legend()
axes[0, 1].legend()

axes[0, 0].grid(linestyle='-')
axes[0, 1].grid(linestyle='-')

axes[0, 1].invert_yaxis()
plt.show()

print('Done and done.')

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os.path
from imageio.core import CannotReadFrameError
from recordclass import recordclass
from scipy import signal
import pickle

import colortracker
import tracker
import utils
import validator
from colortracker import ColorTracker
from footupdown import estimate_detrend
from gait_argument_parser import parse_arguments
from gui import set_threshold
from utils import load_groundtruth
from visualize_gait import visualize_gait

args = parse_arguments()

plt.ioff()

TrackerResults = recordclass('TrackerResults', ['tracker', 'detections', 'tracks'])

# Load videostream
# video_stream = load_video() | stream_from_webcam()
detections_filename = 'TrackerResults/' + args.filename + '.detections.npy'
tracks_filename = 'TrackerResults/' + args.filename + '.tracks.npy'
trackerList = []

if args.cached and os.path.isfile(detections_filename):
    loaded_detections = np.load(detections_filename)
    trackerList = [TrackerResults(tracker=None, detections=detections, tracks=[]) for detections in loaded_detections]
    number_frames = len(loaded_detections[0])
else:
    video_reader = imageio.get_reader('input-videos/' + args.filename)
    number_frames = video_reader.get_meta_data()['nframes']

    # Initialize stuff
    # Select marker/-less
    # Maybe prompt user to select colors, regions, etc.
    # Create instances of necessary classes (SimpleBlobDetector, TrackerKCF, etc.)

    for i in range(args.numOfTrackers):
        # Check if there are any saved values for thresholds and load it
        try:
            with open('hsv-threshold-settings/threshold' + str(i) + '.pkl','rb') as f:
                default_thresholds = pickle.load(f)
        except FileNotFoundError:
            default_thresholds = None

        keypoint_tracker = ColorTracker()
        thresholds = set_threshold(video_reader, default_thresholds)
        keypoint_tracker.hsv_min, keypoint_tracker.hsv_max = thresholds

        trackerList.append(TrackerResults(tracker=keypoint_tracker, detections=[], tracks=[]))

        # Save threshold settings
        with open('hsv-threshold-settings/threshold' + str(i) + '.pkl','wb') as f:
            pickle.dump(thresholds, f)

    font = cv2.FONT_HERSHEY_TRIPLEX

    paused = False

    # Detect keypoints (heel, toe) in each frame

    for frame_nr, img_rgb in enumerate(video_reader):
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        #cv2.putText(img, str(frame_nr), (10, 30), fontFace=font, fontScale=1, color=(0, 0, 255))

        for trackerResult in trackerList:
            detections_for_frame = trackerResult.tracker.detect(img, frame_nr, visualize = True)
            trackerResult.detections.append(detections_for_frame)

        if paused:
            delay = 0
        else:
            delay = 1

        pressed_key = cv2.waitKey(delay) & 0xFF
        if pressed_key == ord(' '):
            paused = not paused
        elif pressed_key == ord('q'):
            break

    for trackerResult in trackerList:
        trackerResult.tracker.cleanup_windows()
        missing_frames = max(number_frames - len(trackerResult.detections), 0)
        trackerResult.detections += [[]] * missing_frames
    np.save(detections_filename, [trackerResult.detections for trackerResult in trackerList])

# Associate keypoints to form tracks

tracks = []

for trackerResult in trackerList:
    trackerResult.tracks = tracker.points_to_tracks(trackerResult.detections,
                                         dist_fun=colortracker.feature_distance(hue_weight=2, 
                                                                                size_weight=2,
                                                                                time_weight=1),
                                         similarity_threshold=140)
    tracks += trackerResult.tracks

np.save(tracks_filename, tracks)

# TODO(rolf): make this plotting code more pretty

fig, axes = plt.subplots(ncols=2, sharex=True)

for track in tracks:
    t_c = [state.frame for state in track.state_history]
    x_c = [state.x for state in track.state_history]
    y_c = [state.y for state in track.state_history]
    axes[0].plot(t_c, signal.detrend(x_c), 'o-', markersize=2)
    axes[1].plot(t_c, signal.detrend(y_c), 'o-', markersize=2)

axes[1].invert_yaxis()
plt.show()

# TODO(rolf): link the subplots in some way to easily see which points correspond,
# for example by highlighting the same x value in both subplots when hovering a point in one subplot

# Generate foot down/up

updown_estimations, x_derivatives = estimate_detrend(tracks, max_frame=number_frames)

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

updown_groundtruth = None
filename_base = os.path.splitext(args.filename)[0]
groundtruth_filename = 'annotations/' + filename_base + '-up_down.npy'

if os.path.isfile(groundtruth_filename):
    updown_groundtruth = load_groundtruth(groundtruth_filename)

    xleftfoot = axes[0, 0].plot(3000 * updown_groundtruth[0, :], 'o-', markersize=2,
                                label='ground truth up/down, left foot')
    yleftfoot = axes[0, 1].plot(750 - 300 * updown_groundtruth[0, :], 'o-', markersize=2,
                                label='ground truth up/down, left foot')
    xrightfoot = axes[0, 0].plot(3000 * updown_groundtruth[1, :], 'o-', markersize=2,
                                 label='ground truth up/down, right foot')
    yrightfoot = axes[0, 1].plot(750 - 300 * updown_groundtruth[1, :], 'o-', markersize=2,
                                 label='ground truth up/down, right foot')
else:
    print('WARNING: could not find ground truth for foot up/down')

axes[0, 0].legend()
axes[0, 1].legend()

axes[0, 0].grid(linestyle='-')
axes[0, 1].grid(linestyle='-')
axes[1, 0].grid(linestyle='-')

axes[0, 1].invert_yaxis()
plt.show()


# Validation

num_groundtruth_tracks = updown_groundtruth.shape[0]
num_estimated_tracks = len(updown_estimations)
errors = np.zeros((num_groundtruth_tracks, num_estimated_tracks))

if updown_groundtruth is not None:
    for row, groundtruth in enumerate(updown_groundtruth):
        for col, estimation in enumerate(updown_estimations):
            errors[row, col] = validator.error(groundtruth, estimation, 0.1)

    matches = utils.greedy_similarity_match(errors, similarity_threshold=0.3)

    ordered_groundtruth = []
    ordered_estimations = []
    matched_estimation_indices = []
    for groundtruth_index, estimation_index in matches:
        ordered_groundtruth.append(updown_groundtruth[groundtruth_index])
        ordered_estimations.append(updown_estimations[estimation_index])
        matched_estimation_indices.append(estimation_index)

    unmatched_estimation_indices = list(set(range(len(updown_estimations))) - set(matched_estimation_indices))
    ordered_estimations += [updown_estimations[index] for index in unmatched_estimation_indices]

    gait_cycle_fig = visualize_gait(ordered_groundtruth, color='green', offset=-1, label='Ground truth')
    visualize_gait(ordered_estimations, fig=gait_cycle_fig, label='Estimated')
else:
    gait_cycle_fig = visualize_gait(updown_estimations, label='Estimated')

gait_cycle_fig.show()
gait_cycle_fig.gca().legend()
plt.show()

print('Done and done.')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os.path
from recordclass import recordclass
from scipy import signal
import pickle

import colortracker
import tracker
import utils
import validator
import stride_parameters
from colortracker import ColorTracker
from footupdown import estimate_detrend
from gait_argument_parser import parse_arguments
from gui import set_threshold
from utils import load_groundtruth
from visualize_gait import visualize_gait

utils.create_necessary_dirs(['hsv-threshold-settings', 
                             'annotations', 
                             'input-videos',
                             'output-videos',
                             'output-data',
                             'TrackerResults'])

args = parse_arguments()

plt.ioff()


detections_filename = 'TrackerResults/' + args.filename + '.detections.pkl'
tracks_filename = 'TrackerResults/' + args.filename + '.tracks.pkl'

# Cache checks
if args.cached and os.path.isfile(detections_filename):
    # loaded_detections = np.load(detections_filename)
    with open(detections_filename, 'rb') as f:
        detections = pickle.load(f)
else:
    detections = colortracker.detect(args.filename, number_of_trackers=args.numOfTrackers)

    with open(detections_filename, 'wb') as f:
        pickle.dump(detections, f)

number_frames = len(detections[0])

# +----------------------------------------------------------------------------+
# |                   Associate keypoints to form tracks                       |
# +----------------------------------------------------------------------------+
tracks = []
for detection_tracker in detections:
    tracks += tracker.points_to_tracks(detection_tracker,
                                       dist_fun=colortracker.feature_distance(hue_weight=2,
                                                                              size_weight=2,
                                                                              time_weight=1),
                                       similarity_threshold=140)

with open(tracks_filename, 'wb') as f:
    pickle.dump(tracks, f)


# TODO(kevin): Find information in tracks to deduce left/right foot and direction they are walking in.
# TODO(kevin): This info will be used in up/down estimations and plotting later on.
# TODO(rolf): make this plotting code prettier

# +----------------------------------------------------------------------------+
# |                Plot the detrended coordinates. 1x2 subplots                |
# +----------------------------------------------------------------------------+ 
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

# +----------------------------------------------------------------------------+
# |                  Generate foot down/up, get derivatives                    |
# +----------------------------------------------------------------------------+ 
updown_estimations, x_derivatives = estimate_detrend(tracks, max_frame=number_frames)

# +----------------------------------------------------------------------------+
# |                              Present results                               |
# +----------------------------------------------------------------------------+ 
f, axes = plt.subplots(ncols=2, nrows=2, sharex=True)
# Add up/down estimations and derivatives in plots. 2x2 subplots
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


# If it exists, add ground truth up/down to the subplots.
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

# Style choices
axes[0, 0].legend()
axes[0, 1].legend()

axes[0, 0].grid(linestyle='-')
axes[0, 1].grid(linestyle='-')
axes[1, 0].grid(linestyle='-')

axes[0, 1].invert_yaxis()
plt.show()


# Validation
# Better visualization of up/down estimation compared to ground truth. 1x1 plot
num_groundtruth_tracks = updown_groundtruth.shape[0]
num_estimated_tracks = len(updown_estimations)
errors = np.zeros((num_groundtruth_tracks, num_estimated_tracks))

# If we have ground truth, sort both lists according to which match the best before visualizing
if updown_groundtruth is not None:
    for row, groundtruth in enumerate(updown_groundtruth):
        for col, estimation in enumerate(updown_estimations):
            errors[row, col] = validator.error(groundtruth, estimation, 0.1)

    # Find similarity pairs
    matches = utils.greedy_similarity_match(errors, similarity_threshold=0.3)

    # Build new lists with the correct order for each pair in 'matches'
    ordered_groundtruth = []
    ordered_estimations = []
    matched_estimation_indices = []
    for groundtruth_index, estimation_index in matches:
        ordered_groundtruth.append(updown_groundtruth[groundtruth_index])
        ordered_estimations.append(updown_estimations[estimation_index])
        matched_estimation_indices.append(estimation_index)

    # Find the set of estimations without a ground truth correspondance and add them last in the list
    unmatched_estimation_indices = list(set(range(len(updown_estimations))) - set(matched_estimation_indices))
    ordered_estimations += [updown_estimations[index] for index in unmatched_estimation_indices]

    # Visualize it
    gait_cycle_fig = visualize_gait(ordered_groundtruth, color='green', offset=-1, label='Ground truth')
    visualize_gait(ordered_estimations, fig=gait_cycle_fig, label='Estimated')
else:
    # Visualize without ground truth
    gait_cycle_fig = visualize_gait(updown_estimations, label='Estimated')

gait_cycle_fig.show()
gait_cycle_fig.gca().legend()
plt.show()

# +----------------------------------------------------------------------------+
# |                     Write stride results to file                           |
# +----------------------------------------------------------------------------+ 
dc = stride_parameters.write_gait_parameters_to_file('output-data/gait.yaml', updown_estimations)

print('Done and done.')

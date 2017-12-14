import os

import numpy as np
import matplotlib.pyplot as plt

from utils import load_updown_groundtruth


def visualize_gait(updown_estimations, args):
    # Better visualization of up/down estimation compared to ground truth. 1x1 plot
    if '%04d' in args.filename:
        video_name = os.path.split(args.filename)[1].split('_%04d')[0]
    else:
        video_name = os.path.splitext(args.filename)[0]

    groundtruth_filename = 'annotations/' + video_name + '-up_down.npy'

    if os.path.isfile(groundtruth_filename):
        updown_groundtruth = load_updown_groundtruth(groundtruth_filename)
        num_groundtruth_tracks = updown_groundtruth.shape[0]
    else:
        num_groundtruth_tracks = 0

    num_estimated_tracks = len(updown_estimations)
    errors = np.zeros((num_groundtruth_tracks, num_estimated_tracks))
    # If we have ground truth, sort both lists according to which match the best before visualizing
    if updown_groundtruth is not None:
        gait_cycle_fig = visualize_updown(updown_groundtruth, color='green', offset=-1, label='Ground truth')
        visualize_updown(updown_estimations, fig=gait_cycle_fig, label='Estimated')
    else:
        # Visualize without ground truth
        gait_cycle_fig = visualize_updown.visualize_gait(updown_estimations, label='Estimated')

    gait_cycle_fig.show()
    gait_cycle_fig.gca().legend()
    plt.show()

def visualize_updown(gait, fig=None, color='black', offset=0, label=None):
    n_gaits = len(gait)
    n_frames = len(gait[0])
    gait_diff = np.diff(gait)

    gait_cycle = []
    for track_index, track in enumerate(gait_diff):
        start = None
        end = None
        gait_cycle.append([])
        # Extract up and down indices
        for frame, chg in enumerate(track):
            if chg == -1:
                start = frame
            if chg == 1:
                if not start: continue
                end = frame
                gait_cycle[track_index].append((start, end-start))

    if fig:
        ax = fig.gca()
    else:
        fig, ax = plt.subplots()
    ax.set_xlim(0, n_frames)
    ax.set_ylim(0, 2*n_gaits+1)
    ax.set_xlabel('Frame')
    ax.grid(linestyle='-')

    for cycle_index, cycle in enumerate(gait_cycle):
        ax.broken_barh(cycle, (offset + 2*cycle_index+1, 1), facecolors=color, label=label)
        label = None # Don't make multiple legend entries with the same content

    return fig

def plot_detrended_coordinates(tracks, signal):
    # +----------------------------------------------------------------------------+
    # |                Plot the detrended coordinates. 1x2 subplots                |
    # +----------------------------------------------------------------------------+ 
    fig, axes = plt.subplots(ncols=2, sharex=True, figsize=(10,5)) # <-- storlek ges i tum. rofl lmao
    for track in tracks:
        t_c = [state.frame for state in track.state_history]
        x_c = [state.x for state in track.state_history]
        y_c = [state.y for state in track.state_history]
        axes[0].plot(t_c, signal.detrend(x_c), 'o-', markersize=2)
        axes[1].plot(t_c, signal.detrend(y_c), 'o-', markersize=2)

    axes[1].invert_yaxis()
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Detrended position x')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Detrended position y')

    plt.show()

    # TODO(rolf): link the subplots in some way to easily see which points correspond,
    # for example by highlighting the same x value in both subplots when hovering a point in one subplot

def present_results(updown_estimations, x_derivatives, used_updown_indexes, tracks, number_frames, args):
    # +----------------------------------------------------------------------------+
    # |                              Present results                               |
    # +----------------------------------------------------------------------------+ 
    f, axes = plt.subplots(ncols=2, nrows=2, sharex=True)
    # Add up/down estimations and derivatives in plots. 2x2 subplots
    # for track_index, point_track in enumerate(tracks):
    for estimate_index, track_index in enumerate(used_updown_indexes):
        track_index = int(track_index)
        point_track = tracks[track_index]
        updown_estimation = updown_estimations[estimate_index]
        estdxline = axes[0, 0].plot((1 + track_index) * 1000 * updown_estimation, 'o-', markersize=2,
                                    label='estimated up/down, index ' + str(track_index))
        estdyline = axes[0, 1].plot(750 - (1 + track_index) * 100 * updown_estimation, 'o-', markersize=2,
                                    label='estimated up/down, index ' + str(track_index))
        derivline = axes[1, 0].plot(range(0, number_frames), x_derivatives[estimate_index], 'o-', markersize=2)

        t = [state.frame for state in point_track.state_history]
        x = [state.x for state in point_track.state_history]

        xline = axes[0, 0].plot(t, x, 'o-', markersize=2, label='x position, index ' + str(track_index))


    # If it exists, add ground truth up/down to the subplots.
    updown_groundtruth = None

    if '%04d' in args.filename:
        video_name = os.path.split(args.filename)[1].split('_%04d')[0]
    else:
        video_name = os.path.splitext(args.filename)[0]
    groundtruth_filename = 'annotations/' + video_name + '-up_down.npy'

    if os.path.isfile(groundtruth_filename):
        updown_groundtruth = load_updown_groundtruth(groundtruth_filename)

        axes[0, 0].plot(3000 * updown_groundtruth[0, :], 'o-', markersize=2,
                                    label='ground truth up/down, left foot')
        axes[0, 1].plot(750 - 300 * updown_groundtruth[0, :], 'o-', markersize=2,
                                    label='ground truth up/down, left foot')
        axes[0, 0].plot(3000 * updown_groundtruth[1, :], 'o-', markersize=2,
                                     label='ground truth up/down, right foot')
        axes[0, 1].plot(750 - 300 * updown_groundtruth[1, :], 'o-', markersize=2,
                                     label='ground truth up/down, right foot')
    else:
        print('WARNING: could not find ground truth for foot up/down')


    axes[0,0].set_xlabel('Frame')
    axes[0,0].set_ylabel('')
    axes[0,1].set_xlabel('Frame')
    axes[0,1].set_ylabel('')

    # Style choices
    axes[0, 0].legend()
    axes[0, 1].legend()

    axes[0, 0].grid(linestyle='-')
    axes[0, 1].grid(linestyle='-')
    axes[1, 0].grid(linestyle='-')

    axes[0, 1].invert_yaxis()
    plt.show()


    


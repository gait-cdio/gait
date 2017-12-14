import numpy as np
import os.path
from scipy import signal
import pickle

import colortracker
import tracker
import utils
import stride_parameters
from footupdown import estimate_detrend
import visualize_gait
from startmenu import start_menu
import subprocess as sp
from openpose_parser import load_ankles_allframes
from footleftright import left_foot_right_foot, left_foot_right_foot_openpose
from utils import load_updown_groundtruth
import validator


def gait_analysis(args, visualize = False):
    if '%04d' in args.filename:
        video_name = os.path.split(args.filename)[1].split('_%04d')[0]
        directory = 'input-images/' + video_name + '/'
    else:
        video_name = os.path.splitext(args.filename)[0]
        directory = 'input-videos/'

    video_path = os.path.join(directory, args.filename)

    detections_filename = 'TrackerResults/' + args.filename + '.detections.pkl'
    tracks_filename = 'TrackerResults/' + args.filename + '.tracks.pkl'

    if args.cached and os.path.isfile(detections_filename):
        # loaded_detections = np.load(detections_filename)
        with open(detections_filename, 'rb') as f:
            detections = pickle.load(f)
    elif args.method == 'markerless':
        sp.call("./openpose_run.sh " + video_path, shell=True)
        detections = load_ankles_allframes('openpose-data/' + video_path)
    else:
        detections = colortracker.detect(video_path,
                                         number_of_trackers=args.numOfTrackers,
                                         set_thresholds=args.set_thresholds)

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

    if args.method == 'markerless':
        dir_and_foot_pairs = left_foot_right_foot_openpose(tracks)
    else:
        dir_and_foot_pairs = left_foot_right_foot(tracks)

    # +----------------------------------------------------------------------------+
    # |                  Generate foot down/up, get derivatives                    |
    # +----------------------------------------------------------------------------+ 
    updown_estimations, x_derivatives, used_updown_indexes = estimate_detrend(tracks, dir_and_foot_pairs, max_frame=number_frames)

    # +----------------------------------------------------------------------------+
    # |            Calculate validation score if ground truth exists               |
    # +----------------------------------------------------------------------------+ 
    groundtruth_filename = 'annotations/' + video_name + '-up_down.npy'

    if os.path.isfile(groundtruth_filename):
        updown_groundtruth = load_updown_groundtruth(groundtruth_filename)
        score = validator.validate(updown_groundtruth, updown_estimations)

    # +----------------------------------------------------------------------------+
    # |                     Write stride results to file                           |
    # +----------------------------------------------------------------------------+
    # Save tracks
    with open(tracks_filename, 'wb') as f:
        pickle.dump(tracks, f)

    # Save updown estimations
    np.save("output-data/" + video_name + "_updown.npy",updown_estimations)

    # Save stride parameters
    dc = stride_parameters.write_gait_parameters_to_file('output-data/{}.yaml'.format(video_name), updown_estimations,
                                                         tracks[0].fps)

    # +----------------------------------------------------------------------------+
    # |                           Visualize results                                |
    # +----------------------------------------------------------------------------+
    if(visualize):
        visualize_gait.plot_detrended_coordinates(tracks, signal)
        visualize_gait.present_results(updown_estimations, x_derivatives, used_updown_indexes, tracks, number_frames, args)
        visualize_gait.visualize_gait(updown_estimations, args)

    return score


if __name__ == "__main__":
    utils.create_necessary_dirs(['hsv-threshold-settings', 
                                 'annotations', 
                                 'input-videos',
                                 'output-videos',
                                 'output-data',
                                 'TrackerResults',
                                 'openpose-data'])

    # args = parse_arguments()
    args = start_menu()
    # plt.ioff()

    gait_analysis(args, visualize = True)

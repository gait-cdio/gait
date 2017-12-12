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
import stride_parameters
from footupdown import estimate_detrend
from utils import load_updown_groundtruth
import visualize_gait
from startmenu import start_menu
import subprocess as sp
from openpose_parser import load_ankles_allframes
from footleftright import left_foot_right_foot
from utils import Direction

def gait_analysis(args, visualize = False):
    detections_filename = 'TrackerResults/' + args.filename + '.detections.pkl'
    tracks_filename = 'TrackerResults/' + args.filename + '.tracks.pkl'

    if args.cached and os.path.isfile(detections_filename):
        # loaded_detections = np.load(detections_filename)
        with open(detections_filename, 'rb') as f:
            detections = pickle.load(f)
    elif args.method == 'markerless':
        sp.call("./openpose_run.sh " + args.filename, shell=True)
        detections = load_ankles_allframes('openpose-data/' + args.filename)
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


    dir_and_foot_pairs = left_foot_right_foot(tracks)

    # +----------------------------------------------------------------------------+
    # |                  Generate foot down/up, get derivatives                    |
    # +----------------------------------------------------------------------------+ 
    updown_estimations, x_derivatives, used_updown_indexes = estimate_detrend(tracks, dir_and_foot_pairs, max_frame=number_frames)
    
    # +----------------------------------------------------------------------------+
    # |                     Write stride results to file                           |
    # +----------------------------------------------------------------------------+
    # Save tracks
    with open(tracks_filename, 'wb') as f:
        pickle.dump(tracks, f)

    # Save updown estimations
    filename_base = os.path.splitext(args.filename)[0]
    np.save("output-data/" + filename_base + "_updown.npy",updown_estimations)

    # Save stride parameters
    dc = stride_parameters.write_gait_parameters_to_file('output-data/{}.yaml'.format(filename_base), updown_estimations,
                                                         tracks[0].fps)

    # +----------------------------------------------------------------------------+
    # |                           Visualize results                                |
    # +----------------------------------------------------------------------------+
    if(visualize):
        visualize_gait.plot_detrended_coordinates(tracks, signal)
        visualize_gait.present_results(updown_estimations, x_derivatives, used_updown_indexes, tracks, number_frames, args)
        visualize_gait.visualize_gait(updown_estimations, args)

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
    plt.ioff()

    gait_analysis(args)

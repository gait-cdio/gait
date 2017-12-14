import os
import argparse
from collections import namedtuple
import yaml

from gait_analysis import gait_analysis


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run validation on all videos under path')
    parser.add_argument('--path', required=True)
    parser.add_argument('--cached', action='store_true')
    args = parser.parse_args()

    return args


def gait_validation(args):
    scores = {}
    methods = ["marker", "markerless"]
    for method in methods:
        scores[method] = {}

    with open('validation-files.yaml', 'rb') as f:
        evaluations_files = yaml.load(f)
    for file_path in evaluations_files.keys():

        path, file = os.path.split(file_path)
        remove_tags = ['_%04d.jpg', '.mp4']
        filename = file
        for tag in remove_tags:
            filename = filename.replace(tag, '')

        groundtruth_filename = filename + "-up_down.npy"
        if os.path.isfile(os.path.join("annotations", groundtruth_filename)):
            gait_args = namedtuple("args",
                                   "filename cached numOfTrackers method set_thresholds")

            gait_args.filename = file
            gait_args.cached = args.cached
            gait_args.set_thresholds = False

            for method in evaluations_files[file_path].keys():
                print("Validating gait analysis on " + filename + " using the " + method + " method...")
                gait_args.method = method
                if method == 'marker':
                    gait_args.numOfTrackers = evaluations_files[file_path][method]['num_of_trackers']
                scores[method][filename] = gait_analysis(gait_args)

        else:
            print("Warning: Ground truth for " + filename + " is missing.")

    with open(os.path.join("output-data", 'validation-scores.yaml'), 'w') as f:
        yaml.dump(scores, f, default_flow_style=False)


if __name__ == "__main__":
    if True:
        openpose = {'markerless': 'openpose'}
        markers2 = {'marker': {'num_of_trackers': 2}, 'markerless': 'openpose'}
        markers4 = {'marker': {'num_of_trackers': 4}, 'markerless': 'openpose'}
        evaluations_files = {
            'input-videos/4farger.mp4': markers4,
            'input-videos/4fargersilly.mp4': markers4,
            # 'input-videos/4markerskevin1.mp4': markers2,
            'input-videos/4markerskevin2.mp4': markers2,
            'input-videos/4markersjohn1.mp4': markers2,
            'input-videos/4markersjohn3.mp4': markers2,
            'input-images/john_markerless/john_markerless_%04d.jpg': openpose,
            'input-images/rolf_markerless/rolf_markerless_%04d.jpg': openpose,
            #'input-images/kevin_markerless/kevin_markerless_%04d.jpg': openpose,
        }

        with open('validation-files.yaml', 'w') as f:
            yaml.dump(evaluations_files, f, default_flow_style=False)

    args = parse_arguments()
    gait_validation(args)

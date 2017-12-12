import os
import argparse
from collections import namedtuple

from gait_analysis import gait_analysis

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run validation on all videos under path')
    parser.add_argument('--path', required=True)
    parser.add_argument('--cached', default=False, required=False)
    args = parser.parse_args()

    return args

def gait_validation(args):
    for filename in sorted(os.listdir(args.path)):
        if filename.endswith(".mp4"):
            groundtruth_filename = os.path.splitext(filename)[0] + "-up_down.npy"
            if(os.path.isfile(os.path.join("annotations", groundtruth_filename))):
                gait_args = namedtuple("args", 
                                       "filename cached numOfTrackers method set_thresholds")

                gait_args.filename = filename
                gait_args.cached = args.cached
                gait_args.numOfTrackers = 4
                gait_args.set_thresholds = False

                for method in ["marker", "markerless"]:
                    print("Validating gait analysis on " + filename + " using the " + method + " method...")
                    gait_args.method = method
                    gait_analysis(gait_args, visualize=True)
                
            else:
                print("Warning: Ground truth for " + filename + " is missing.")

if __name__ == "__main__":
    args = parse_arguments()
    gait_validation(args)

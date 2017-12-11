import json

import numpy as np
import os
import pickle

from colortracker import PointFeatures


def load_ankles_frame(filename):
    with open(filename, 'r') as f:
        frame = json.load(f)

    pose_keypoints = frame['people'][0]['pose_keypoints']
    left_ankle = pose_keypoints[3*13:3*14]
    right_ankle = pose_keypoints[3*10:3*11]
    return left_ankle, right_ankle

def load_ankles_allframes(foldername):
    detections_left = []
    detections_right = []
    for index, filename in enumerate(sorted(os.listdir(foldername))):
        left_ankle, right_ankle = load_ankles_frame(os.path.join(foldername, filename))
        detections_left.append([PointFeatures(
            position=left_ankle[0:2],
            size=5,
            hue=0,
            frame=index
        )])
        detections_right.append([PointFeatures(
            position=right_ankle[0:2],
            size=5,
            hue=0,
            frame=index
        )])
    return [detections_left, detections_right]

# load_keypoints('openpose-data/4farger/4farger_000000000038_keypoints.json')
all_ankles = load_ankles_allframes('openpose-data/4farger')
np.save('openpose-data/4farger.detections.openpose.npy', all_ankles)
with open('ankles.pkl', 'wb') as f:
    pickle.dump(all_ankles, f)

with open('ankles.pkl', 'rb') as f:
    all_ankles_loaded = pickle.load(f)
# all_ankles_loaded = np.load('openpose-data/4farger.detections.openpose.npy')

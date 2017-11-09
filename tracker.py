""" A track that is following a constant velocity model """

import numpy as np
from collections import namedtuple

from utils import greedy_similarity_match

PositionData = namedtuple('PositionData', ['x', 'y', 'frame'])

def clamp(n, smallest, largest): return max(smallest, min(n, largest))

class Track:
    def __init__(self, start_frame=None, x=None, y=None, dx=0, dy=0, fps=30, feature=None):
        # State space model
        # x = Fx + bu
        # y = Hx + du

        self.dt = 1.0/fps

        self.x = np.matrix([[x], [dx], [y], [dy]], dtype=float)

        self.F = np.eye(4, dtype=float) # F = 1 dt  0  0
        self.F[0, 1] = self.dt          #     0  1  0  0
        self.F[2, 3] = self.dt          #     0  0  1 dt
                                        #     0  0  0  1
                                        #
        self.H = np.matrix([[1,0,0,0],  # H = 1  0  0  0
                            [0,0,1,0]]) #     0  0  1  0

        # A priori covariance
        self.P = np.asmatrix(np.diag([16, 10000, 16, 10000]), dtype=float)
        self.Q = np.asmatrix(np.diag([16, 10000, 16, 10000]), dtype=float)
        self.R = np.asmatrix(np.diag([4, 4]), dtype=float)

        # Feature used for distance calculation
        self.feature = feature

        self.current_frame = self.start_frame = start_frame
        self.state_history = []
        self.score = 1

    def update_feature(self, feature = None):
        if feature: self.feature = feature
        self.feature.position = (self.x[0,0], self.x[2,0])

    def measurement_update(self, position):
        # Kalman filter measurement update
        z = np.matrix([[position[0]],[position[1]]], dtype=float)
        x = self.x; F = self.F; H = self.H; P = self.P; R = self.R

        y = z - H*x
        S = R + H*P*H.transpose()
        K = P*H.transpose()*np.linalg.inv(S)

        I = np.eye(4, dtype=float)
        self.x = x + K*y
        self.P = (I - K*H)*P

    def predict(self):
        x = self.x; F = self.F; P = self.P; Q = self.Q

        old_position = x[0, 0], x[2, 0]
        self.x = F*x
        self.P = F*P*F.transpose() + Q
        self.state_history.append(PositionData(x=old_position[0], y=old_position[1], frame=self.current_frame))

        self.current_frame += 1


def points_to_tracks(detections, dist_fun, similarity_threshold=10000):
    tracks = []
# Run Kalman filter
    for frame, new_detections in enumerate(detections):
        # Prediction 
        for track in tracks:
            # Only keep tracking if score is sufficient
            if track.score > 0:
                track.predict()
                track.update_feature()

        # If tracks is empty, create new track for each detection
        tracked_detections = [t.feature for t in tracks]
        detection_scores = [t.score for t in tracks]
        match_list = match(detection_scores, tracked_detections, new_detections, dist_fun, similarity_threshold)

        for track_index, detection_index in match_list:
            tracks[track_index].measurement_update(new_detections[detection_index].position)
            tracks[track_index].update_feature(new_detections[detection_index])

        # TODO(rolf): make this readable
        not_found  = list(set(range(0, len(tracks))) -         set([t[0] for t in match_list]))
        new_tracks = list(set(range(0, len(new_detections))) - set([t[1] for t in match_list]))

        # Update score for tracks
        for track_index, track in enumerate(tracks):
            if track_index in not_found:
                track.score -= 1
            else:
                track.score += 1
            # TODO: Clamp score, this is tweakable
            track.score = clamp(track.score, -2, 8)

        # Add new tracks
        for new in new_tracks:
            x, y = new_detections[new].position
            tracks.append(Track(start_frame=frame, x=x, y=y, feature=new_detections[new]))

    return tracks


def match(track_scores, tracks, detections, dist_fun, similarity_threshold):
    # Create similarity matrix
    sim_mat = np.zeros((len(tracks), len(detections)))
    for i, e1 in enumerate(tracks):
        for j, e2 in enumerate(detections):
            # If score is invalid, set dist to 'inf' to not match
            if track_scores[i] <= 0:
                sim_mat[i, j] = np.inf
            else:
                sim_mat[i, j] = dist_fun(e1, e2)

    return greedy_similarity_match(sim_mat, similarity_threshold)


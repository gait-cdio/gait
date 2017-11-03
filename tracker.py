""" A track that is following a constant velocity model """

import numpy as np

class Track:
    def __init__(self, x = 0, y = 0, dx = 0, dy = 0, fps = 30, feature = None):
        # State space model
        # x = Ax + bu
        # y = cx + du

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
        self.P = np.asmatrix(np.diag([0.05, 0.05, 0.5, 0.5]), dtype=float)
        self.Q = np.asmatrix(np.diag([0.01, 0.01, 0.1, 0.1]), dtype=float)
        self.R = np.asmatrix(np.diag([0.01, 0.01]), dtype=float)

        # Feature used for distance calculation
        self.feature = feature

    def update_feature(self, feature = None):
        if(feature): self.feature = feature
        self.feature.position = (self.x[0,0], self.x[2,0])

    def correct(self, x, y):
        # Kalman filter measurement update
        z = np.matrix([[x],[y]], dtype=float)
        x = self.x; F = self.F; H = self.H; P = self.P; R = self.R

        y = z - H*x
        S = R + H*P*H.transpose()
        K = P*H.transpose()*np.linalg.inv(S)

        I = np.eye(4, dtype = float)
        self.x = x + K*y
        self.P = (I - K*H)*P
    
    def predict(self):
        x = self.x; F = self.F; P = self.P; Q = self.Q

        self.x = F*x
        self.P = F*P*F.transpose() + Q


def points_to_tracks(detections, dist_fun, similarity_threshold=10000):
    tracks = []
# Run kalman filter
    for new_detections in detections:
        # Prediction 
        for track in tracks:
            track.predict()
            track.update_feature()

        # If tracks is empty, create new track for each detection
        tracked_detections = [t.feature for t in tracks]
        match_list = match(tracked_detections, new_detections, dist_fun, similarity_threshold)

        for track_index, detection_index in match_list:
            x, y = new_detections[detection_index].position
            tracks[track_index].correct(x, y)
            tracks[track_index].update_feature(new_detections[detection_index])

        not_found  = list(set(range(0,len(tracks))) -         set([t[0] for t in match_list]))
        new_tracks = list(set(range(0,len(new_detections))) - set([t[1] for t in match_list]))

        for new in new_tracks:
            x, y = new_detections[new].position
            tracks.append(Track(x, y, feature=new_detections[new]))

    return tracks


def match(list1, list2, dist_fun, similarity_threshold):
    # Create similarity matrix
    sim_mat = np.zeros((len(list1), len(list2)))
    for i, e1 in enumerate(list1):
        for j, e2 in enumerate(list2):
            sim_mat[i, j] = dist_fun(e1, e2)

    # Get match list greedy by always picking the minimum distance
    match_list = []
    while True:
        try:
            best_match = np.unravel_index(np.nanargmin(sim_mat), (len(list1), len(list2)))
        except ValueError:
            break

        if sim_mat[best_match] < similarity_threshold:
            match_list.append(best_match)
            
        sim_mat[:, best_match[1]] = np.nan
        sim_mat[best_match[0], :] = np.nan

    return match_list

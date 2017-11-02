""" A track that is following a constant velocity model """

import numpy as np

class Track:
    def __init__(self, x = 0, y = 0, dx = 0, dy = 0, fps = 30):
        # State space model
        # x = Ax + bu
        # y = cx + du

        self.dt = 1.0/fps

        self.x = np.matrix([[x], [y], [dx], [dy]], dtype=float)

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

    def update(self, x, y):
        # Kalman filter measurement update
        z = np.matrix([[x],[y]], dtype = float)
        x = self.x; F = self.F; H = self.H; P = self.P; R = self.R

        y = z - H*x
        S = R + H*P*H.transpose()
        K = P*H.transpose()*np.linalg.inv(S)

        I = np.eye(4, dtype = float)
        self.x = x + K*y
        self.P = (I - K*H)*P
    
    def predict(self):
        x = self.x; F = self.F; P = self.P; Q = self.Q;

        self.x = F*x
        self.P = F*P*F.transpose() + Q


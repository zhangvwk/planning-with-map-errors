import numpy as np


class LQRPlanner:
    # def __init__(self, Q, R, A, B):
    #     self.Q = Q
    #     self.R = R
    #     self.A = A
    #     self.B = B

    def __init__(self):
        pass

    def compute_path(self, u, v):
        cost = np.linalg.norm(u - v)
        path = np.vstack((u, v))
        return cost, path

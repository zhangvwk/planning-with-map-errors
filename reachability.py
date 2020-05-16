# Standard libraries
import numpy as np

# Custom libraries
from utils import TestUtil


class ReachabilityFilter:
    def __init__(self):
        self.step = 0
        self.initialize_coeffs()

    def set_controller_gain(self, K):
        self.K = K

    def store_regulator_terms(self):
        self.BK = self.B.dot(self.K)
        self.reg_matrix = self.A - self.BK

    def update_step(self):
        self.step += 1

    def set_motion_model(self, A, B, Q):
        TestUtil.check_square(A)
        TestUtil.check_symmetric(Q)
        self.A = A
        self.B = B
        self.Q = Q

    def set_obs_model(self, R):
        self.R = R

    def update_obs_matrix(self, env, lines_seen):
        self.C = None

    def get_state_zonotope(self):
        if self.step == 0:
            self.a = self.reg_matrix.dot(self.a)
            self.e = np.eye(self.A.shape[0])
            self.b = self.reg_matrix.dot(self.b) - self.BK.dot(self.e)
            self.c = [np.eye(self.A.shape[0])]
            self.d = [0]
            self.p = []
            self.q = []
        else:
            est_matrix = np.eye(self.A.shape[0]) - self.L.dot(self.C)
            est_matrix_A = est_matrix.dot(self.A)
            self.a = self.reg_matrix.dot(self.a)
            self.e = est_matrix_A.dot(self.e)
            self.b = self.reg_matrix.dot(self.b) - self.BK.dot(self.e)
            self.p.append(-est_matrix)
            self.q.append(self.L)
            for n in range(self.step):
                self.c[n] = self.reg_matrix.dot(self.c[n]) - self.BK.dot(self.p[n])
                self.d[n] = self.reg_matrix.dot(self.d[n]) - self.BK.dot(self.q[n])
                self.p[n] = est_matrix.dot(self.p[n])
                self.q[n] = est_matrix.dot(self.q[n])
            self.c.append(np.eye(self.A.shape[0]))
            self.d.append(0)
        self.update_step()

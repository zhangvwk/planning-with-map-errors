# Standard libraries
import numpy as np

# Custom libraries
from search import PlanUtils


class Simulator:
    """Simulator class for rolling out trajectories given nominal waypoints and inputs.
    """

    def __init__(self, env):
        self.env = env

    def set_motion(self, A, B, Q):
        self.A = A
        self.B = B
        self.Q = Q

    def set_obs(self, R):
        self.R = R

    def set_gain(self, K):
        self.K = K

    def set_init(self, x0):
        self.x0 = x0

    def set_init_est(self, x_est_0, P_est_0):
        self.x_est_0 = x_est_0
        self.P_est_0 = P_est_0

    def simulate_state(self, x, u):
        return self.A.dot(x) + self.B.dot(u) + np.random.normal(0, self.Q)

    def get_obs_matrices(self, x, conf_factor=0.5):
        lines_seen_now = PlanUtils.get_lines_seen_now(self.env, x[:2])
        C, b, e = PlanUtils.get_observation_matrices(
            lines_seen_now, self.env, len(x), actual_err=True
        )
        Rhat = PlanUtils.get_Rhat(self.R, e, conf_factor)
        return C, b, Rhat

    def simulate_obs(self, x, C, b, Rhat):
        return self.C.dot(x) + b + np.random.normal(0, Rhat)

    def get_controls(self, u_nom, x_est, x_nom):
        return u_nom - self.K.dot(x_est - x_nom)

    def predict(self, x_est, u, P_est):
        x_bar = self.A.dot(x_est) + u
        P_bar = (self.A.T.dot(P_est)).dot(self.A.T) + self.Q
        return x_bar, P_bar

    def get_kalman_gain(self, P_bar, C, Rhat):
        return (P_bar.dot(C.T)).dot(np.linalg.inv((C.dot(P_bar)).dot(C.T) + Rhat))

    def innovate(self, x_bar, L, z, C, P_bar):
        x_est = x_bar + L.dot(z - C.dot(x_bar))
        P_est = (np.eye(L.shape[0]) - L.dot(C)).dot(P_bar)
        return x_est, P_est

    def rollout(self, x0, x_noms, u_noms):
        x = x0
        xs = [x]
        x_est = self.x_est_0
        P_est = self.P_est_0
        for k in range(len(x_noms)):
            u = self.update_controls(u_noms[k], x_est, x_noms[k])
            x_bar, P_bar = self.predict(x_est, u, P_est)
            x = self.simulate_state(x, u)
            C, b, Rhat = self.get_obs_matrices(x)
            z = self.simulate_obs(x, C, b, Rhat)
            L = self.get_kalman_gain(P_bar, C, Rhat)
            x_est, P_est = self.innovate(x_bar, L, z, C, P_bar)
            xs.append(x)
        return x

    def run(self, iters, x0s, x_noms, u_noms):
        xs = {}
        for i in range(iters):
            xs[i] = self.rollout(x0s[i], x_noms, u_noms)
        return xs

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom libraries
from utils import PlotTools
from plan import PlanUtils
from shapes import Point


class Simulator:
    """Simulator class for rolling out trajectories given nominal waypoints and inputs.
    """

    def __init__(self, env):
        self.env = env
        self.A = None
        self.B = None
        self.Q = None
        self.R = None
        self.K = None
        self.C = None
        self.x_est_0 = None
        self.P_est_0 = None
        self.motion_ready = False
        self.obs_ready = False
        self.gain_ready = False
        self.est_ready = False

    def set_motion(self, A, B, Q):
        self.A = A
        self.B = B
        self.Q = Q
        self.motion_ready = True

    def set_obs(self, R):
        self.R = R
        self.obs_ready = True

    def set_gain(self, K):
        self.K = K
        self.gain_ready = True

    def set_init_est(self, x_est_0, P_est_0):
        self.x_est_0 = x_est_0
        self.P_est_0 = P_est_0
        self.est_ready = True

    def is_initialized(self):
        return (
            self.motion_ready and self.obs_ready and self.gain_ready and self.est_ready
        )

    def simulate_state(self, x, u, scale=1):
        return (
            self.A.dot(x)
            + self.B.dot(u)
            + np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q) * scale
        )

    def get_obs_matrices(self, x, conf_factor=3):
        lines_seen_now = PlanUtils.get_lines_seen_now(
            self.env, Point(x[0], x[1]), config="actual"
        )
        C, b_actual, b_half, e = PlanUtils.get_observation_matrices(
            lines_seen_now, self.env, len(x), actual_err=True
        )
        Rhat = PlanUtils.get_Rhat(self.R, e, conf_factor)
        return C, b_actual, b_half, e, Rhat

    def simulate_obs(self, x, C, b):
        z = C.dot(x) + b
        return z + np.random.multivariate_normal(
            np.zeros(C.shape[0]), self.R * np.eye(C.shape[0])
        )

    def get_controls(self, u_nom, x_est, x_nom):
        return u_nom - self.K.dot(x_est - x_nom)

    def predict(self, x_est, u, P_est):
        x_bar = self.A.dot(x_est) + self.B.dot(u)
        P_bar = (self.A.dot(P_est)).dot(self.A.T) + self.Q
        return x_bar, P_bar

    def get_kalman_gain(self, P_bar, C, Rhat):
        return (P_bar.dot(C.T)).dot(np.linalg.inv((C.dot(P_bar)).dot(C.T) + Rhat))

    def innovate(self, x_bar, L, z, C, b_half, P_bar):
        x_est = x_bar + L.dot(z - C.dot(x_bar) - b_half)
        P_est = (np.eye(L.shape[0]) - L.dot(C)).dot(P_bar)
        return x_est, P_est

    def rollout(self, x0, x_noms, u_noms, scales=None):
        assert self.is_initialized()
        x = x0
        xs = [x]
        x_est = self.x_est_0
        P_est = self.P_est_0
        x_ests = [x_est]
        x_bars = []
        if scales is None:
            scales = np.ones(len(x_noms))
        collision = False
        for k in range(len(x_noms)):
            u = self.get_controls(u_noms[k], x_est, x_noms[k])
            x_bar, P_bar = self.predict(x_est, u, P_est)
            x = self.simulate_state(x, u, scales[k])
            if self.env.contains(Point(x[0], x[1]), config="actual"):
                collision = True
            C, b_actual, b_half, e, Rhat = self.get_obs_matrices(x)
            z = self.simulate_obs(x, C, b_actual)
            try:
                L = self.get_kalman_gain(P_bar, C, Rhat)
                x_est, P_est = self.innovate(x_bar, L, z, C, b_half, P_bar)
            except:
                x_est, P_est = x_bar, P_bar
            xs.append(x)
            x_ests.append(x_est)
            x_bars.append(x_bar)
        return xs, x_ests, x_bars, collision

    def run(self, iters, x0, S, x_noms, u_noms, scales=None, verbose=True):
        xs = {}
        x_ests = {}
        x_bars = {}
        num_collisions = 0
        for i in tqdm(range(iters)):
            xs[i], x_ests[i], x_bars[i], collision = self.rollout(
                x0 + np.random.multivariate_normal(np.zeros(S.shape[0]), S),
                x_noms,
                u_noms,
                scales,
            )
            num_collisions += int(collision)
        prob_collision = float(num_collisions / iters)
        return xs, x_ests, x_bars, prob_collision

    def plot_trajs(self, xs, x_noms):
        for i in xs:
            PlotTools.plot_traj(xs[i])
        PlotTools.plot_traj(x_noms, linestyle="--", color="k")

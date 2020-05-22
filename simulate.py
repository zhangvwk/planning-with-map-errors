# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom libraries
from search import PlanUtils
from shapes import Point


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

    def set_init_est(self, x_est_0, P_est_0):
        self.x_est_0 = x_est_0
        self.P_est_0 = P_est_0

    def simulate_state(self, x, u):
        return (
            self.A.dot(x)
            + self.B.dot(u)
            + np.random.multivariate_normal(np.zeros(self.Q.shape[0]), self.Q)
        )

    def get_obs_matrices(self, x, conf_factor=0.5):
        lines_seen_now = PlanUtils.get_lines_seen_now(self.env, Point(x[0], x[1]))
        # print("lines_seen_now = {}".format(lines_seen_now))
        C, b_actual, b_ref, e = PlanUtils.get_observation_matrices(
            lines_seen_now, self.env, len(x), actual_err=True
        )
        Rhat = PlanUtils.get_Rhat(self.R, e, conf_factor)
        return C, b_actual, b_ref, e, Rhat

    def simulate_obs(self, x, C, b, Rhat):
        Rhat = np.array(Rhat)
        z = C.dot(x) + b
        return z + np.random.multivariate_normal(np.zeros(Rhat.shape[0]), Rhat)

    def get_controls(self, u_nom, x_est, x_nom):
        return u_nom - self.K.dot(x_est - x_nom)

    def predict(self, x_est, u, P_est):
        x_bar = self.A.dot(x_est) + self.B.dot(u)
        P_bar = (self.A.T.dot(P_est)).dot(self.A.T) + self.Q
        return x_bar, P_bar

    def get_kalman_gain(self, P_bar, C, Rhat):
        return (P_bar.dot(C.T)).dot(np.linalg.inv((C.dot(P_bar)).dot(C.T) + Rhat))

    def innovate(self, x_bar, L, z, C, b_ref, P_bar):
        x_est = x_bar + L.dot(z - C.dot(x_bar) - b_ref)
        P_est = (np.eye(L.shape[0]) - L.dot(C)).dot(P_bar)
        return x_est, P_est

    def rollout(self, x0, x_noms, u_noms):
        x = x0
        xs = [x]
        x_est = self.x_est_0
        P_est = self.P_est_0
        for k in range(len(x_noms)):
            # print("==========================")
            # print("x = {}".format(x))
            u = self.get_controls(u_noms[k], x_est, x_noms[k])
            # print("u = {}".format(u))
            x_bar, P_bar = self.predict(x_est, u, P_est)
            # print("x_bar, P_bar = {}, {}".format(x_bar, P_bar))
            x = self.simulate_state(x, u)
            # print("x = {}".format(x))
            C, b_actual, b_ref, e, Rhat = self.get_obs_matrices(x)
            # print(
            #     "C, b_actual, b_half, Rhat = {},{},{},{}".format(
            #         C, b_actual, b_half, Rhat
            #     )
            # )
            z = self.simulate_obs(x, C, b_actual, Rhat)
            try:
                L = self.get_kalman_gain(P_bar, C, Rhat)
                x_est, P_est = self.innovate(x_bar, L, z, C, b_ref, P_bar)
            except:
                x_est, P_est = x_bar, P_bar
            # print("x_est, P_est = {},{}".format(x_est, P_est))
            xs.append(x)
        return xs

    def run(self, iters, x0, S, x_noms, u_noms, verbose=True):
        xs = {}
        for i in tqdm(range(iters)):
            xs[i] = self.rollout(
                x0 + np.random.multivariate_normal(np.zeros(S.shape[0]), S),
                x_noms,
                u_noms,
            )
        return xs

    def plot_traj(self, x, linestyle="-", color="r"):
        x_list, y_list = map(list, zip(*[(state[0], state[1]) for state in x]))
        plt.plot(x_list, y_list, linestyle=linestyle, color=color)

    def plot_trajs(self, xs, x_noms):
        for i in xs:
            self.plot_traj(xs[i])
        self.plot_traj(x_noms, linestyle="--", color="k")

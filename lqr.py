# Standard libraries
import numpy as np
from scipy.linalg import solve_discrete_are


class LQRPlanner:
    """Class for computing the optimal trajectory and cost between
    two points, assuming infinite horizon discrete LQR.
    """

    def __init__(self, Q, R, A, B):
        """LQRPlanner constructor

        Args:
            Q: state cost (NxN)
            R: control cost (MxM)
            A: State dynamics (NxN)
            B: Control dynamics (NxM)
        """
        self.Nstate = A.shape[0]
        self.Ncontrol = B.shape[1]
        self.Q = Q
        if self.Ncontrol == 1:
            self.R = np.array(R).reshape((self.Ncontrol,))
        else:
            self.R = R
        self.A = A
        self.B = B
        self.compute_gain()  # compute and store the gain at initialization

    def compute_gain(self):
        """Solve infinite horizon discrete ARE."""
        V = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.gain = np.linalg.inv(self.R + self.B.T.dot(V.dot(self.B)))
        self.gain = self.gain.dot(self.B.T.dot(V.dot(self.A)))

    def get_gain(self):
        """Return the LQR gain.
        Meant to be called outside the class.
        """
        return self.gain

    def compute_path(self, u, v, max_iter=1000, tol=1e-3, get_lqr_cost=False):
        """Compute a path from u to v using an LQR regulator

        Args:
            u: starting point
            v: final point
            max_iter: maximum iteration before stopping forward path
            tol: termination criterion for forward pass
        """
        lqr_cost = 0
        l2_cost = 0
        path = [u - v]
        done = False
        converged = False
        iteration = 0
        # controls = np.empty((0, self.gain.shape[0]))
        controls = []

        while not done:
            x = path[iteration]
            control = -self.gain.dot(x)
            # controls = np.vstack((controls, control))
            controls.append(control)
            lqr_cost += 0.5 * x.T.dot(self.Q.dot(x)) + 0.5 * control.T.dot(
                self.R.dot(control)
            )

            x_new = self.A.dot(x) + self.B.dot(control)
            l2_cost += np.linalg.norm(x_new[:2] - x[:2])
            path.append(x_new)
            iteration += 1

            converged = np.linalg.norm(x_new) < tol
            done = converged or (iteration >= max_iter)

        if not converged:
            lqr_cost = np.Inf
            l2_cost = np.Inf

        Nsteps = len(path)
        path = np.vstack(path).reshape(Nsteps, self.Nstate)
        path += v.reshape(1, self.Nstate)
        controls = np.vstack(controls).reshape(-1, self.Ncontrol)
        if not get_lqr_cost:
            # print("u, v = {}, {}".format(u, v))
            # print("l2_cost = {}".format(l2_cost))
            return l2_cost, path, controls
        return lqr_cost, path, controls


class DummyPlanner:
    def __init__(self):
        pass

    def compute_path(self, u, v):
        cost = np.linalg.norm(u - v)
        path = np.vstack((u, v))
        return cost, path

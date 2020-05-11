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
        Ncontrol = B.shape[1]
        self.Q = Q
        if Ncontrol == 1:
            self.R = np.array(R).reshape((Ncontrol,))
        else:
            self.R = R
        self.A = A
        self.B = B
        self.compute_gain()  # compute and store the gain at initialization

    def compute_gain(self):
        """Solve infinite horizon discrete ARE."""
        V = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.lqr_gain = np.linalg.inv(self.R + self.B.T.dot(V.dot(self.B)))
        self.lqr_gain = self.lqr_gain.dot(self.B.T.dot(V.dot(self.A)))

    def get_gain(self):
        """Return the LQR gain.
        Meant to be called outside the class.
        """
        return self.lqr_gain

    def compute_path(self, u, v, max_iter=1000, tol=1e-3):
        """Compute a path from u to v using an LQR regulator

        Args:
            u: starting point
            v: final point
            max_iter: maximum iteration before stopping forward path
            tol: termination criterion for forward pass
        """
        cost = 0
        path = [u - v]
        done = False
        converged = False
        iteration = 0

        while not done:
            x = path[iteration]
            control = -self.lqr_gain.dot(x)
            cost += 0.5 * x.T.dot(self.Q.dot(x)) + 0.5 * control.T.dot(
                self.R.dot(control)
            )

            x = self.A.dot(x) + self.B.dot(control)
            path.append(x)
            iteration += 1

            converged = np.linalg.norm(x) < tol
            done = converged or (iteration >= max_iter)

        if not converged:
            cost = np.Inf

        Nsteps = len(path)
        Nstate = u.shape[0]
        path = np.vstack(path).reshape(Nsteps, Nstate)
        path += v.reshape(1, Nstate)
        return cost, path


class DummyPlanner:
    def __init__(self):
        pass

    def compute_path(self, u, v):
        cost = np.linalg.norm(u - v)
        path = np.vstack((u, v))
        return cost, path

import numpy as np
from scipy.linalg import solve_discrete_are


class LQRPlanner:
    def __init__(self, Q=0, R=0, A=0, B=np.array([0,0]).reshape(2,1)):
        """LQRPlanner constructor

        Default arguments are meaningless and left here for
        backward compatibility.

        Args:
            Q: state cost (NxN)
            R: control cost (MxM)
            A: State dynamics (NxN)
            B: Control dynamics (NxM)
        """
        Ncontrol = B.shape[1]
        self.Q = Q
        if Ncontrol==1:
            self.R = np.array(R).reshape((Ncontrol,))
        else:
            self.R = R
        self.A = A
        self.B = B


    def compute_path(self, u, v):
        """ Left here for backward compatibility."""
        cost = np.linalg.norm(u - v)
        path = np.vstack((u, v))
        return cost, path


    def compute_path_new(self, u, v, max_iter=1000, tol=1e-6):
        """Compute a path from u to v using an LQR regulator

        Args:
            u: starting point
            v: final point
            max_iter: maximum iteration before stopping forward path
            tol: termination criterion for forward pass
        """
        # solve infinite horizon discrete ARE
        V = solve_discrete_are(self.A, self.B, self.Q, self.R)
        lqr_gain = np.linalg.inv(self.R + self.B.T.dot(V.dot(self.B)))
        lqr_gain = lqr_gain.dot(self.B.T.dot(V.dot(self.A)))

        cost = 0
        path = [u-v]
        done = False
        converged = False
        iteration = 0

        while not done:
            x = path[iteration]
            control = -lqr_gain.dot(x)
            cost += 0.5*x.T.dot(self.Q.dot(x)) + 0.5*control.T.dot(self.R.dot(control))

            x = self.A.dot(x) + self.B.dot(control)
            path.append(x)
            iteration += 1

            converged = np.linalg.norm(x)<tol
            done = converged or (iteration>=max_iter)

        if not converged:
            cost = np.Inf

        Nsteps = len(path)
        Nstate = u.shape[0]
        path =  np.vstack(path).reshape(Nsteps, Nstate)
        path += v.reshape(1, Nstate)
        return cost, path

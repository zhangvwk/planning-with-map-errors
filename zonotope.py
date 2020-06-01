import numpy as np
import polytope as pc
import math
from scipy.special import erf, erfinv
from shapely.geometry import Polygon
from pypoman import compute_polytope_vertices

SCALING_FOR_INCLUSION = np.sqrt(2) * erfinv(0.9 ** 0.5)


def area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


class Zonotope:
    def __init__(self, center, generators, cov):
        self.c = center
        self.G = generators
        self.Sig = cov
        self._H_is_valid = False
        assert self.G.shape[0] == self.c.shape[0]
        assert self.G.shape[0] == self.Sig.shape[0]
        assert self.G.shape[0] == self.Sig.shape[1]

    def get_order(self):
        return self.G.shape[1] / self.G.shape[0]

    def __add__(self, other):
        c = self.c + other.c
        G = np.hstack(
            (self.G.reshape(self.G.shape[0], -1), other.G.reshape(other.G.shape[0], -1))
        )
        Sig = self.Sig + other.Sig
        return Zonotope(c, G, Sig)

    def __le__(self, other):
        """Return True if self is enclosed by other."""
        XY = pc.extreme(self.get_confidence_sets(SCALING_FOR_INCLUSION)[0].to_poly())
        A, b = other.get_confidence_sets(SCALING_FOR_INCLUSION)[0].to_H()
        n_points = XY.shape[0]
        ineq = A.dot(XY.T)
        for i in range(n_points):
            if not np.all(ineq[:, i] < b):
                return False
        return True

    def __str__(self):
        return "({}, {})".format(self.c, self.G)

    def scale(self, T):
        self.c = T.dot(self.c)
        self.G = T.dot(self.G)
        self.Sig = T.dot(self.Sig.dot(T.transpose()))
        self._H_is_valid = False # invalidate the stored H representation
        # print("c = {}".format(self.c))
        # print("T = {}".format(T))
        # print("Sig = {}".format(self.Sig))

    def to_H(self):
        """Convert to an hyper-plane representation of a polytope

        This operation is expensive but tractable when n=2.
        For more details, see Matthias' thesis, p. 15.
        """
        if (self._H_is_valid):
            return self.A, self.b

        if len(self.G.shape) == 1:
            self.G = np.reshape(self.G, (self.G.shape[0], 1))
        n = self.G.shape[0]
        e = self.G.shape[1]
        assert n == 2  # what follows is only valid in 2D
        Cp = np.zeros((e, n))
        dp = np.zeros((e,))
        dm = np.zeros((e,))
        for i in range(e):
            g = self.G[:, i]  # extract generator i
            gg = np.array([g[1], -g[0]])  # compute the cross product nX()
            if np.linalg.norm(gg) < 1e-10:
                Cp[i:,:] = 0
                dp[i] = 1
                dm[i] = 1
                continue
            Cpi = gg / np.linalg.norm(gg)
            delta_di = 0
            for j in range(e):
                delta_di += np.fabs(
                    Cpi.dot(self.G[:, j])
                )  # dot product with all generators

            Cp[i, :] = Cpi
            dp[i] = Cpi.dot(self.c) + delta_di
            dm[i] = -Cpi.dot(self.c) + delta_di

        # store the H-representation and allow to reuse it until it is not valid anymore
        self.A = np.vstack((Cp, -Cp))
        self.b = np.concatenate((dp, dm))
        self._H_is_valid = True
        return self.A, self.b

    def to_poly(self):
        A, b = self.to_H()
        return pc.Polytope(A, b)

    def reduce(self, target_order=1.0):
        """Reduce the zonotope by replacing 4 well chosen generators by 2 generators
        This is the girard method as described in:
        Reachability of Uncertain Linear Systems Using Zonotopes (2005)
        """
        if self.G.shape[1] < 4:
            return

        self._H_is_valid = False # invalidate the stored H-rep

        # keep reducing as long as the order is greater than the target order
        while self.get_order() > target_order:
            # need at least 4 generators
            if self.G.shape[1] < 4:
                return

            norm_diff = np.linalg.norm(self.G, ord=1, axis=0) - np.linalg.norm(
                self.G, ord=np.inf, axis=0
            )
            select = np.argpartition(norm_diff, 4)[:4]

            chosen_g = self.G[:, select]
            coeff = np.sum(np.fabs(chosen_g), axis=1)
            self.G = np.delete(self.G, select, axis=1)
            self.G = np.hstack((self.G, np.array([[coeff[0], 0], [0, coeff[1]]])))

    def get_confidence_sets(self, scaling_factors):
        """
        Compute a list of confidence set associated with the scaling factors.
        Each element of that list is a zonotope with zero covariance,
        ie a deteriminstic zonotope.
        See Matthias' thesis, p. 96.
        """
        # conpute the generators of the G-zonotope
        n = self.G.shape[0]
        vals, vecs = np.linalg.eig(self.Sig)
        gs = np.sqrt(vals) * vecs
        # get the zonotope representation of each confidence set
        confid_sets = []
        scaling_factors = np.atleast_1d(scaling_factors)  # to handle scalar case
        for i in range(scaling_factors.shape[0]):
            m = scaling_factors[i]
            z = Zonotope(np.zeros((n,)), m * gs, np.zeros((n, n))) + self
            z.Sig = np.zeros((n, n))
            confid_sets.append(z)
        return confid_sets

    def get_inter_prob(self, X, scaling_factors):
        """
        Compute intersection probability
        In the notation of Matthias' thesis, the scaling_factors are:
            gamma=m(0), m(1), ..., m(k-1)
        """
        n = self.G.shape[0]
        assert n == 2  # what follows is only valid in 2D
        scaling_factors = np.array(scaling_factors)
        k = scaling_factors.shape[0]

        # calling pc.extreme on this one is fast, maybe because it is a plain reactangle?
        X = Polygon(pc.extreme(X.to_poly()))

        confid_sets = self.get_confidence_sets(scaling_factors)
        # append m=0 to the set of scaling factors
        scaling_factors = np.append(scaling_factors, 0.0)

        h = (
            (2.0 * np.pi) ** (-n / 2.0)
            * np.linalg.det(self.Sig) ** (-0.5)
            * np.exp(-0.5 * np.array(scaling_factors) ** 2.0)
        )

        # compute intersection volumes
        V = np.zeros((k,))
        for i in range(k):
            A, b = confid_sets[i].to_H()
            vertices = compute_polytope_vertices(A, b) # fast C code for vertex enumeration
            vertices.sort(key=lambda c:math.atan2(c[0], c[1])) # sort those vertices
            X2 = Polygon(vertices) # construct a Polygon
            V[i] = X2.intersection(X).area # this is faster than intersect of polytope

        # compute intersection prob
        prob = 1 - erf(scaling_factors[0] / np.sqrt(2.0)) ** (2.0 * n)
        prob += h[0] * V[0]
        for i in range(k):
            prob += (h[i + 1] - h[i]) * V[i]

        return min(prob, 1.0)

    def intersect(self, Z, m):
        """Check the intersection between 2 (probabilistic) zonotopes
        m is the scaling factor for the confidence set
        """
        # These are deterministic zonotopes
        s1 = self.get_confidence_sets(m)[0]
        s2 = Z.get_confidence_sets(m)[0]
        return not pc.is_empty(s1.to_poly().intersect(s2.to_poly()))

    def plot(self, scaling_factors, ax):
        for confid_set in self.get_confidence_sets(scaling_factors):
            confid_set.to_poly().plot(ax=ax)

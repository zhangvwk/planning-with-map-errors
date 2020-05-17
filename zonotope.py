import numpy as np
import polytope as pc
from scipy.special import erf

def area(x, y):
    return 0.5*np.abs( \
           np.dot(x,np.roll(y,1)) - \
           np.dot(y,np.roll(x,1)) \
           )

class Zonotope:
    def __init__(self, center, generators, cov):
        self.c = center
        self.G = generators
        self.Sig = cov
        assert self.G.shape[0] == 2
        assert self.c.shape[0] == 2
        assert self.Sig.shape == (2, 2)

    def get_order(self):
        return self.G.shape[1] / 2

    def __add__(self, other):
        c = self.c + other.c
        G = np.hstack((self.G, other.G.reshape(2, -1)))
        Sig = self.Sig + other.Sig
        return Zonotope(c, G, Sig)

    def scale(self, T):
        self.c = T.dot(self.c)
        self.G = T.dot(self.G)
        self.Sig = T.dot(self.Sig.dot(T.transpose()))

    def to_H(self):
        """Convert to an hyper-plane representation of a polytope

        This operation is expensive but tractable when n=2.
        For more details, see Matthias' thesis, p. 15.
        """
        n = self.G.shape[0]
        e = self.G.shape[1]
        Cp = np.zeros((e, n))
        dp = np.zeros((e,))
        dm = np.zeros((e,))
        for i in range(e):
            g = self.G[:, i]  # extract generator i
            gg = np.array([g[1], -g[0]])  # compute the cross product nX()
            Cpi = gg / np.linalg.norm(gg)
            delta_di = 0
            for j in range(e):
                delta_di += np.fabs(
                    Cpi.dot(self.G[:, j])
                )  # dot product with all generators

            Cp[i, :] = Cpi
            dp[i] = Cpi.dot(self.c) + delta_di
            dm[i] = -Cpi.dot(self.c) + delta_di

        return np.vstack((Cp, -Cp)), np.concatenate((dp, dm))

    def to_poly(self):
        A, b = self.to_H()
        return pc.Polytope(A, b)

    def reduce(self, target_order=1.):
        """Reduce the zonotope by replacing 4 well chosen generators by 2 generators
        This is the girard method as described in:
        Reachability of Uncertain Linear Systems Using Zonotopes (2005)
        """
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
        vals, vecs = np.linalg.eig(self.Sig)
        gs = np.sqrt(vals) * vecs
        # get the zonotope representation of each confidence set
        confid_sets = []
        scaling_factors = np.atleast_1d(scaling_factors) # to handle scalar case
        for i in range(scaling_factors.shape[0]):
            m = scaling_factors[i]
            z = Zonotope(np.zeros((2,)), m*gs, np.zeros((2,2))) + self
            z.Sig = np.zeros((2,2))
            confid_sets.append(z)
        return confid_sets

    def get_inter_prob(self, X, scaling_factors):
        '''
        Compute intersection probability
        In the notation of Matthias' thesis, the scaling_factors are:
            gamma=m(0), m(1), ..., m(k-1)
        '''
        n = 2
        k = scaling_factors.shape[0]
        X = X.to_poly()

        h = (2.*np.pi)**(-n/2.) * np.linalg.det(self.Sig)**(-0.5) * \
            np.exp(-0.5*np.array(scaling_factors)**2.)

        confid_sets = self.get_confidence_sets(scaling_factors)
        V = np.zeros((k,))
        for i in range(k):
            p = confid_sets[i].to_poly().intersect(X)
            XY = pc.extreme(p)
            if XY is not None:
               V[i] = area(XY[:,0], XY[:,1])

        prob = 1 - erf(scaling_factors[0]/np.sqrt(2.)) ** (2.*n) + \
               h[1]*V[0] - h[k-1]*V[k-1]
        for i in range(1,k-1):
            prob += (h[i+1]-h[i]) * V[i]

        return min(prob, 1.)

    def intersect(self, Z, m):
        """Check the intersection between 2 (probabilistic) zonotopes
        m is the scaling factor for the confidence set
        """
        # These are deterministic zonotopes
        s1 = self.get_confidence_sets(m)[0]
        s2 = Z.get_confidence_sets(m)[0]
        return not pc.is_empty(s1.to_poly().intersect(s2.to_poly()))


import numpy as np
import polytope as pc
import matplotlib.pyplot as plt


class Zonotope:
    def __init__(self, center, generators, cov):
        self.c = center
        self.G = generators
        self.Sig = cov
        assert self.G.shape[0]==2
        assert self.c.shape[0]==2
        assert self.Sig.shape==(2,2)

    def __add__(self, other):
        c = self.c + other.c
        G = np.hstack((self.G, other.G.reshape(2,-1)))
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
        Cp = np.zeros((e,n))
        dp = np.zeros((e,))
        dm = np.zeros((e,))
        for i in range(e):
            g = self.G[:,i] # extract generator i
            gg = np.array([g[1], -g[0]]) # compute the cross product nX()
            Cpi = gg / np.linalg.norm(gg)
            delta_di = 0
            for j in range(e):
                delta_di += np.fabs(Cpi.dot(self.G[:,j])) # dot product with all generators

            Cp[i,:] = Cpi
            dp[i] = Cpi.dot(self.c) + delta_di
            dm[i] = -Cpi.dot(self.c) + delta_di

        return np.vstack((Cp, -Cp)), np.concatenate((dp, dm))

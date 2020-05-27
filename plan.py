# Standard libraries
import numpy as np
from bitarray import bitarray, frozenbitarray
from copy import deepcopy

# Custom libraries
from utils import GeoTools
from shapes import Point
from zonotope import Zonotope


class PlanUtils:
    @staticmethod
    def get_lines_seen_now(env, point, config="full"):
        if not isinstance(point, np.ndarray):
            return env.get_lines_seen(point, config=config)
        else:
            return env.get_lines_seen(Point(point[0], point[1]), config=config)

    @staticmethod
    def rectlines2lines(rectlines):
        lines = []
        for rectangle_idx, line_list in rectlines.items():
            for line_idx in line_list:
                lines.append(rectangle_idx * 4 + line_idx)
        return np.array(lines)

    @staticmethod
    def get_observation_matrices(lines_seen_now, env, n, actual_err=False):
        C = np.empty((0, n))
        v = np.zeros(n)
        b_half = []
        b_actual = []
        w = []
        for rectangle_idx, lines in lines_seen_now.items():
            rectangle = env.rectangles[rectangle_idx]
            for line_idx in lines:
                line = rectangle.edges["full"][line_idx]
                v[:2] = GeoTools.get_normal_vect(line)
                mid_point = GeoTools.get_mid_point(line).as_array()
                center_point = rectangle.get_center_gen()[0]
                # Make sure v is pointed outward wrt rectangle surface
                if (mid_point - center_point).dot(v) < 0:
                    v = -v
                C = np.vstack((C, v))
                half_wdth = rectangle.half_widths[line_idx]
                if actual_err:
                    b_actual.append(
                        -(mid_point + rectangle.actual_errors[line_idx] * v[:2]).dot(
                            v[:2]
                        )
                    )
                b_half.append(
                    -(mid_point + rectangle.center_offsets[line_idx] * v[:2]).dot(v[:2])
                )
                w.append(half_wdth)
        if actual_err:
            return C, np.array(b_actual), np.array(b_half), np.array(w)
        else:
            return C, np.array(w)

    @staticmethod
    def get_dist(C, b, e, state):
        return C.dot(state) + b + e

    @staticmethod
    def get_Rhat(R, w, conf_factor):
        m = len(w)  # number of measurements
        Rhat = np.zeros((m, m))
        for i in range(m):
            Rhat[i, i] = ((w[i] / conf_factor) + np.sqrt(R)) ** 2
        return Rhat

    @staticmethod
    def init_bitarray(nlines_tot):
        return bitarray(nlines_tot)

    @staticmethod
    def configID2bitarray(configID, Sn, nlines_tot):
        n_extr = len(Sn)
        bitstring_format = "{" + "0:0%db" % n_extr + "}"
        b = bitarray(bitstring_format.format(configID)[::-1])
        bconfig = PlanUtils.init_bitarray(nlines_tot)
        bconfig.setall(False)
        for i in range(n_extr):
            if b[i]:
                bconfig[Sn[i]] = True
        return bconfig


class NuValues:
    def __init__(self, Sn, w, cov, nlines_tot):
        """
        Construct an object given a numpy array of line IDs.
        This object is entitled to that set of line IDs.
        Ideally I would store that line but can I afford duplicate storage?
        """
        assert Sn.size < nlines_tot

        # should never change, number of lines seen at n
        self._nlines = Sn.size
        self._values = {}
        self._w = w
        self._cov = cov
        self._nlines_tot = nlines_tot

        self._bitmask = PlanUtils.init_bitarray(nlines_tot)
        self._bitmask.setall(False)
        # set the bits corresponding to the lines seen to 1
        for i in range(self._nlines):
            self._bitmask[Sn[i]] = True

        self.set_values(Sn, Sn)

    def set_values(self, Sn, Sk):
        """
        Set values from the same Sn used for the constructor and Sk.
        w is the half-width of the uncertain mean.
        R is the variance for all measurements.
        """
        assert Sn.size == self._nlines  # incomplete check but still useful
        self._values = {}  # reinitialize

        idx_extr = np.in1d(Sn, Sk)  # True for elements in Sn also in Sk, size of Sn
        # number of lines that need to be considered as extrema, those are both in Sn and Sk
        n_extr = np.where(idx_extr)[0].size

        n_config = 2 ** n_extr

        # bits is used to keep track which extremum has been considered
        # for each line considered as extrema
        bits = bitarray(n_extr)
        bitstring_format = "{" + "0:0%db" % n_extr + "}"

        centers = np.zeros((Sn.size,))
        generators = np.zeros((Sn.size,))

        # final key of size the total number of lines in the environment
        key = PlanUtils.init_bitarray(self._nlines_tot)

        for i_config in range(n_config):
            # set a bitmask for the current config in consideration
            bits = bitarray(bitstring_format.format(i_config)[::-1])

            centers.fill(0)
            generators.fill(0)
            key.setall(False)

            i = 0
            for iSn in range(Sn.size):
                if idx_extr[iSn]:
                    # a line in Sn that is also in Sk
                    if bits[i]:
                        # choose + half width
                        centers[iSn] = self._w[iSn]
                        key[Sn[iSn]] = True  # Sn[iSn] = this line ID
                    else:
                        # choose - half width
                        centers[iSn] = -self._w[iSn]
                    i += 1
                else:
                    # a line in Sn that is not in Sk, consider the full range
                    generators[iSn] = self._w[iSn]

            # all extrema must have been considered
            assert i == n_extr
            self._values[frozenbitarray(key)] = deepcopy(
                Zonotope(centers, generators, self._cov * np.eye(Sn.size))
            )

    def at_config(self, Sk, configID):
        """
        Return the zonotope associated with a config ID.
        configID can be between 0 and 2^|S_k|-1.
        Can only be used with the most recent Sk.
        """
        n_extr = len(Sk)
        assert configID < 2 ** n_extr

        bconfig = PlanUtils.configID2bitarray(configID, Sk, self._nlines_tot)
        # filter it by the bitmask of lines seen at n and return
        return self._values[frozenbitarray(self._bitmask & bconfig)]


class Plan:
    def __init__(self, start_point, start_idx, env, n, R=0.1, kmax=100):
        self.head = start_point
        self.path = [start_point]
        self.cost = 0
        self.head_idx = start_idx

        self.R = R  # 1D distance measurement variance
        self.n = n
        self.kmax = kmax
        self.k = 0
        self._nlines_tot = env.nlines_tot

        self.initialize_variables()
        self.initialize_Nu(start_point, env)

    def update_head_idx(self, head_idx):
        self.head_idx = head_idx

    def initialize_variables(self):
        # Coefficients that do not require the entire history
        self.a = np.eye(self.n)
        self.b = np.zeros((self.n, self.n))
        self.e = np.eye(self.n)

        # Coefficients that require the entire history whose shape does not change
        self.c = np.zeros((self.n, self.n, self.kmax))
        self.c[:, :, 0] = np.eye(2)
        self.p = np.zeros((self.n, self.n, self.kmax))

        # same as above but changing shapes
        self.d = None
        self.q = None

        # Estimation matrices
        self.A = None
        self.B = None
        self.Q = None
        self.K = None
        self.C = None
        self.P = None
        self.L = None

        # Initialization flags
        self.motion_ready = False
        self.gain_ready = False
        self.est_ready = False

    def initialize_Nu(self, start_point, env):
        lines_seen_now = PlanUtils.get_lines_seen_now(env, start_point)
        self.rectangles_seen_now = lines_seen_now.keys()
        line_indices = PlanUtils.rectlines2lines(lines_seen_now)
        _, w = PlanUtils.get_observation_matrices(lines_seen_now, env, self.n)
        self.Sn = [line_indices]
        Nu_new = NuValues(self.Sn[0], w, self.R, self._nlines_tot)
        self.Nu = [Nu_new]

        # initialize Nu_full
        self.Nu_full = [Zonotope(np.zeros(w.shape), w, self.R * np.eye(w.shape[0]))]

    def set_motion(self, A, B, Q):
        self.A = A
        self.B = B
        self.Q = Q
        self.motion_ready = True

    def set_gain(self, K):
        self.K = K
        self.gain_ready = True

    def set_init_est(self, P0):
        self.P0 = P0
        self.P = P0
        self.est_ready = True

    def is_initialized(self):
        return self.motion_ready and self.gain_ready and self.est_ready

    def update_info(self, head_idx, cost_to_add, path_to_add):
        self.head_idx = head_idx
        self.cost += cost_to_add
        self.path_to_add = np.vstack((self.path, path_to_add))

    def add_point(self, env, point, conf_factor=0.5):
        """
        - get C(k+1), R(k+1)
        - P = Pbar - L*C*Pbar
        - Pbar = A*P*A.T + Q
        - L = Eq (1.9)
        - update all Nu(n) if needed
        """
        if self.k == 0:
            assert self.is_initialized()
        self.head = point
        lines_seen_now = PlanUtils.get_lines_seen_now(env, point)
        self.rectangles_seen_now = lines_seen_now.keys()
        line_indices = PlanUtils.rectlines2lines(lines_seen_now)
        self.C, w = PlanUtils.get_observation_matrices(lines_seen_now, env, self.n)
        Pbar = (self.A.dot(self.P)).dot(self.A.T) + self.Q
        Rhat = PlanUtils.get_Rhat(self.R, w, conf_factor)
        self.L = (Pbar.dot(self.C.T)).dot(
            np.linalg.inv((self.C.dot(Pbar)).dot(self.C.T)) + Rhat
        )
        self.P = Pbar - (self.L.dot(self.C)).dot(Pbar)

        self.update_Nu(env, line_indices, w)
        self.update_coeffs()

    def update_Nu(self, env, lines, w):
        self.Sn.append(lines)
        prev_lines = self.Sn[self.k]  # self.k not updated yet
        same = np.intersect1d(lines, prev_lines)
        additional = np.setdiff1d(lines, same)
        missing = np.setdiff1d(prev_lines, same)

        # if S(k) = S(k+1), no need to update anything
        if additional.size > 0 or missing.size > 0:
            for n in range(self.k + 1):
                if (
                    np.intersect1d(additional, self.Sn[n]).size == 0
                    or np.intersect1d(missing, self.Sn[n]).size == 0
                ):
                    self.Nu[n].set_values(self.Sn[n], self.Sn[self.k + 1])

        Nu_new = NuValues(self.Sn[self.k + 1], w, self.R, self._nlines_tot)
        self.Nu.append(Nu_new)
        self.Nu_full.append(Zonotope(np.zeros(w.shape), w, self.R * np.eye(w.shape[0])))
        self.k += 1

    def update_coeffs(self):
        """
        - from k to k+1
        - assume L and C are already update by add_point
        """
        M1 = self.A - self.B.dot(self.K)
        M2 = np.eye(self.n) - self.L.dot(self.C)

        self.a = M1.dot(self.a)
        self.b = M1.dot(self.b) - self.B.dot(self.K.dot(self.e))
        self.e = M2.dot(self.A.dot(self.e))

        self.c[:, :, self.k] = np.eye(2)
        self.p[:, :, self.k] = -M2

        m = self.L.shape[1]
        d_next = np.zeros((self.n, m, self.k+1))
        q_next = np.zeros((self.n, m, self.k+1))
        q_next[:, :, self.k] = self.L

        # n = 1 ... self.k-1
        for n in range(1, self.k):
            self.c[:, :, n] = M1.dot(self.c[:, :, n]) - self.B.dot(
                self.K.dot(self.p[:, :, n])
            )
            d_next[:, :, n] = M1.dot(self.d[:, :, n]) - self.B.dot(
                self.K.dot(self.q[:, :, n])
            )
            self.p[:, :, n] = M2.dot(self.A.dot(self.p[:, :, n]))
            q_next[:, :, n] = M2.dot(self.A.dot(self.q[:, :, n]))

        self.d = d_next
        self.q = q_next

    def get_max_prob_collision(self, env):
        """
        - called after add_point and update
        - now we have everything to compute all possible X(k+1)
          and intersect those with the associated environment
        """
        n_extr = len(self.Sn[-1])
        Xks = self.get_Xks()
        p = 0.0
        for configID in range(2 ** n_extr):
            bconfig = self.configID2bitarray(configID, self.Sn[-1], self._nlines_tot)
            for rectangle_idx in self.rectangles_seen_now:
                config = bconfig[rectangle_idx * 4 : (rectangle_idx + 1) * 4]
                p = max(
                    p,
                    self.get_prob_collision(
                        Xks[configID],
                        env.rectangles[rectangle_idx].to_zonotope(config),
                    ),
                )
        return p

    def get_prob_collision(self, Xk, rectangle_zono):
        """Return the probability of collision between the reachable set
        Xk and a Rectangle object.
        """
        return Xk.get_inter_prob(rectangle_zono)

    def get_Xks(self):
        """Return a dictionary of format:
        {configId -> corresponding reachable set Xk}
        """
        amb = self.a - self.b
        center_offset = self.head + amb.dot(self.path[0])
        cov_offset = amb.dot(self.P0.dot(amb.T))
        for n in range(1, self.k + 1):
            cov_offset += self.c[:, :, n].dot(self.Q.dot(self.c[:, :, n].T))

        n_extr = len(self.Sn[-1])
        Xks = {}
        for configID in range(2 ** n_extr):
            # trick because I don't have a zero zonotope
            Xks[configID] = self.Nu[1].at_config(self.Sn[-1], configID)
            Xks[configID].scale(self.d[:, :, 1])
            for n in range(2, self.k + 1):
                Xks[configID] += (
                    self.Nu[n].at_config(self.Sn[-1], configID).scale(self.d[:, :, n])
                )
            Xks[configID].c += center_offset
            Xks[configID].Sig += cov_offset
        return Xks

# Standard libraries
import numpy as np
import collections
from bitarray import bitarray, frozenbitarray

# Custom libraries
from utils import GeoTools
from zonotope import Zonotope


class NuValues:
    NLINES_TOT = -1 #  to initialize in the driver code

    def __init__(self, Sn):
        '''
        Construct an object given a numpy array of line IDs.
        This object is entitled to that set of line IDs.
        Ideally I would store that line but can I afford duplicate storage?
        '''
        assert Sn.shape[0] < NLINES_TOT

        # should never change, number of lines seen at n
        self._nlines = Sn.size
        self._values = {}

        self._bitmask = bitarray([False for i in range(N_LINES_TOT)])
        # set the bits corresponding to the lines seen to 1
        for i in range(self._nlines):
            self._bitmask[Sn[i]] = True


    def set_values(Sn, Sk, w, R):
        '''
        Set values from the same Sn used for the constructor and Sk.
        w is the half-width of the uncertain mean.
        R is the variance for all measurements.
        '''
        idx_extr = np.in1d(Sn, Sk) # True for elements in Sn also in Sk
        n_extr = np.where(idx_extr)[0].size # number of lines that need to be considered as extrema

        n_config = 2**n_extr
        n_measures = w.size
        assert n_measures == Sn.size

        bits = bitarray(n_extr)
        centers = np.zeros((n_measures,))
        generators = np.zeros((n_measures,))
        covariances = np.zeros((n_measures,))

        key = bitarray(N_LINES_TOT)

        for i_config in range(n_config):
            # set a bitmask for the current config in consideration
            bits.setall(False)
            bits += bitarray(format(i_config, 'b'))

            centers.fill(0)
            generators.fill(0)
            key.setall(False)

            # counter for the config bitmask
            i = 0
            for i_measure in range(n_measures):
                if idx_extr[i_measure]:
                    # choose + or - the halfwidth
                    if bits[i]:
                        centers[i_measure] = e[i_measure]
                        key[Sn[i_measure]] = True
                    else:
                        centers[i_measure] = -e[i_measure]
                    i += 1
                else:
                    # consider the full range
                    generators[i_measure] = e[i_measure]

            # all extrema must have been considered
            assert i == n_extr
            values[frozenbitarray(key)] = Zonotope(centers, generators, R*np.eye(n_measures))


    def at_config(Sk, configID):
        '''
        Return the zonotope associated with a config ID.
        configID can be between 0 and 2^|S_k|-1.
        '''
        b = bitarray(format(configID, 'b'))
        bconfig = bitarray([False for i in range(N_LINES_TOT)])
        for i in range(Sk.shape[0]):
            if b[i]:
                bconfig[Sk[i]] = True
        # filter it by the bitmask of lines seen at n and return
        return self.values[frozenbitarray(self._bitmask & bconfig)]


class Plan:
    def __init__(self, head_idx, path, cost, zonotope_state):
        self.head_idx = head_idx
        self.path = path
        self.cost = cost
        self.zonotope_state = zonotope_state

        self.n = 2  # replace this by taking the dimension from the arguments
        self.kmax = 100  # an argument with a default value?
        self.k = 0

        # coefficients that do not require the entire history
        self.a = np.eye(self.n)
        self.b = np.zeros(self.n)
        self.e = np.eye(self.n)

        # coefficients that require the entire history
        self.c = np.zeros((self.n, self.n, self.kmax))
        self.c[:, :, 0] = np.eye(2)
        self.d = np.zeros((self.n, self.n, self.kmax))
        self.d[:, :, 0] = np.zeros(2)
        self.p = np.zeros((self.n, self.n, self.kmax))
        self.q = np.zeros((self.n, self.n, self.kmax))

        # Kalman matrices
        self.C = None
        self.P = None
        self.Pbar = None
        self.L = None

        # for Nu stuff
        # TODO construct from the initial point
        self.Sn = []
        self.Nu = []

        # Lines
        self.lines_seen_now = None
        self.lines_seen_tot = None


    def add_point(self, env, point, A, B, Q, R, conf_factor=0.5):
        """
        - get C(k+1), R(k+1)
        - P = Pbar - L*C*Pbar
        - Pbar = A*P*A.T + Q
        - L = Eq (1.9)
        - update all Nu(n) if needed
        """
        lines_seen_now = PlanUtils.get_lines_seen_now(env, point)
        line_indices = PlanUtils.rectlines2lines(lines_seen_now)
        PlanUtils.update_lines_seen_tot(lines_seen_now, self.lines_seen_tot)
        self.C, self.b, self.e, line_ids = PlanUtils.get_observation_matrices(
            lines_seen_now, env, self.n
        )
        self.Pbar = (A.dot(self.P)).dot(A.T) + Q
        self.Rhat = PlanUtils.get_Rhat(R, self.e, conf_factor)
        self.L = (self.Pbar.dot(self.C.T)).dot(
            np.linalg.inv((self.C.dot(self.Pbar)).dot(self.C.T)) + self.Rhat
        )
        self.P = self.Pbar - (self.L.dot(self.C)).dot(self.Pbar)

        self.update_Nu(env, lines)


    def update_Nu(self, env, lines):
        self.Sn.append(lines)
        prev_lines = self.Sn[self.k] # self.k not updated yet
        same = np.intersect1d(lines, prev_lines)
        additional = np.setdiff1d(lines, same)
        missing = np.setdiff1d(prev_lines, same)

        # if S(k) = S(k+1), no need to update anything
        if additional.size>0 or missing.size>0:
            for n in range(self.k+1):
                if np.intersect1d(additional, self.Sn[n]).size == 0 or \
                   np.intersect1d(missing,    self.Sn[n]).size == 0:
                    Nu_new = NuValues(Sn[n])
                    Nu_new.set_values(Sn[n], Sn[self.k+1], self.e)
                    self.Nu[n] = Nu_new

        Nu_new = NuValues(Sn[self.k+1])
        Nu_new.set_values(Sn[self.k+1], Sn[self.k+1], self.e)
        self.Nu.append(Nu_new)
        self.k += 1


    def update(self, A, B, K):
        """
        - from k to k+1
        - assume L and C are already update by add_point
        """
        M1 = A - B.dot(K)
        M2 = np.eye(self.n) - self.L.dot(self.C)

        self.a = M1.dot(self.a)
        self.b = M1.dot(self.b) - B.dot(K.dot(self.e))
        self.e = M2.dot(A.dot(self.e))

        self.c[:, :, self.k] = np.eye(2)
        self.p[:, :, self.k] = -M2
        self.q[:, :, self.k] = self.L

        # n = 0 ... self.k-1
        for n in range(self.k):
            self.c[:, :, n] = M1.dot(self.c[:, :, n]) - B.dot(K.dot(self.p[:, :, n]))
            self.d[:, :, n] = M1.dot(self.d[:, :, n]) - B.dot(K.dot(self.q[:, :, n]))
            self.p[:, :, n] = M2.dot(A.dot(self.p[:, :, n]))
            self.q[:, :, n] = M2.dot(A.dot(self.q[:, :, n]))


    def get_prob_unsafe(self, env):
        """
        - called after add_point and update
        - now we have everything to compute all possible X(k+1)
          and intersect those with the associated environment
        """
        pass


class PlanUtils:
    @staticmethod
    def get_lines_seen_now(env, point):
        return env.get_lines_seen(point)

    @staticmethod
    def update_lines_seen_tot(lines_seen_now, lines_seen_tot):
        for k, v in lines_seen_now.items():
            if k in lines_seen_tot:
                lines_seen_tot[k] = list(set(lines_seen_tot[k] + v))
            else:
                lines_seen_tot[k] = v

    @staticmethod
    def rectlines2lines(rectlines):
        lines = []
        for rectangle_idx, line_list in rectlines.items():
            for line_idx in line_list:
                lines.add(rectangle_idx * 4 + line_idx)
        return np.array(lines)

    @staticmethod
    def get_observation_matrices(lines_seen_now, env, n, actual_err=False):
        C = np.empty((0, n))
        v = np.zeros(n)
        b_half = []
        b_actual = []
        # b_ref = []
        e = []
        line_ids = []
        gens = np.empty((0, 2))
        for rectangle_idx, lines in lines_seen_now.items():
            rectangle = env.rectangles[rectangle_idx]
            # print("===== rectangle_idx = {} =====".format(rectangle_idx))
            for line_idx in lines:
                # print("=== line_idx = {} ===".format(line_idx))
                line = rectangle.edges[line_idx]
                v[:2] = GeoTools.get_normal_vect(line)
                mid_point = GeoTools.get_mid_point(line).as_array()
                center_point = rectangle.get_center_gen()[0]
                # Make sure v is pointed outward wrt rectangle surface
                if (mid_point - center_point).dot(v) < 0:
                    v = -v
                C = np.vstack((C, v))
                bound_l, bound_r = rectangle.error_bounds[line_idx]
                # print("bound_l, bound_r = {}, {}".format(bound_l, bound_r))
                half_wdth = (bound_r - bound_l) / 2
                if actual_err:
                    # print("mid_point = {}".format(mid_point))
                    b_actual.append(
                        -(mid_point + rectangle.actual_errors[line_idx] * v[:2]).dot(
                            v[:2]
                        )
                    )
                # b_ref.append(-(mid_point.dot(v[:2])))
                b_half.append(
                    -(mid_point + (-abs(bound_l) + half_wdth) * v[:2]).dot(v[:2])
                )
                gens = np.vstack((gens, half_wdth * v[:2]))
                e.append(half_wdth)
                line_ids.append(rectangle_idx * 4 + line_idx)
        if actual_err:
            return C, np.array(b_actual), np.array(b_half), np.array(e)
        else:
            return C, np.array(b_half), np.array(e), np.array(line_ids)

    @staticmethod
    def get_dist(C, b, e, state):
        return C.dot(state) + b + e

    @staticmethod
    def get_Rhat(R, e, conf_factor):
        m = len(e)  # number of measurements
        Rhat = np.zeros((m, m))
        for i in range(m):
            Rhat[i, i] = ((e[i] / conf_factor) + np.sqrt(R)) ** 2
            # Rhat[i, i] = R
        return Rhat


class Searcher:
    def __init__(self, graph, pruning_coeff=0.5):
        self.graph = graph
        self.pruning_coeff = pruning_coeff
        self.x_init = None
        self.goal_region = None
        self.P_open = set()
        self.P = collections.defaultdict(set)
        self.G = set()

    def set_source(self, x_init=None):
        if not x_init:
            self.x_init = self.graph.samples[0, :]
        else:
            self.x_init = x_init
        self.initialize_open()

    def initialize_open(self):
        self.P_open.add(Plan(self.x_init, None, 0, None))
        self.G = self.P_open

    def set_goal(self, goal_region):
        self.goal_region = goal_region

    def is_valid_goal(self, verbose=True):
        return GeoTools.is_valid_region(self.goal_region, self.graph.env, verbose)

    def reached_goal(self):
        return len([p for p in self.G if self.in_goal(p[0])]) != 0

    def in_goal(self, point):
        return self.goal_region.A.dot(point) <= self.goal.b

    def remove_dominated(self):
        for p in self.P_open:
            for q in self.P:
                if q.head != p.head:
                    continue
                else:
                    lower_cost = q.cost < p.cost
                    enclosed = q.zonotope_state <= p.zonotope_state
                    if lower_cost and enclosed:
                        self.P_open.remove(p)
                        self.P.remove(p)

    def prune(self, i):
        for p in self.P_open:
            if p.cost <= i * self.pruning_coeff:
                self.G.add(p)

    def explore(self):
        if self.x_init is None:
            print("Setting the source node as the first node in the graph.")
            self.set_source()
        try:
            assert self.is_valid_goal()
        except AssertionError:
            print("Goal region not defined.")
            raise
        i = 0
        while not self.P_open and not self.reached_goal():
            for p in self.G:
                for k, v in self.graph.edges[p.head_idx].items():
                    discard = False
                    neighbor_idx = k
                    to_neighbor_cost, to_neighbor_path = v
                    zonotope_state_sub = p.zonotope_state
                    for sub_neighbor in to_neighbor_path:
                        zonotope_state_sub = self.reachability_filter.propagate(
                            sub_neighbor, zonotope_state_sub
                        )  # add lines seen?
                        prob_collision = self.reachability_filter.get_probability(
                            self.graph, zonotope_state_sub
                        )
                        if self.collision(prob_collision):
                            break
                            discard = True
                    if discard:
                        continue
                    q = Plan(
                        neighbor_idx,
                        np.vstack((p.path, to_neighbor_path)),
                        p.cost + to_neighbor_cost,
                        zonotope_state_sub,
                    )
                    self.P[neighbor_idx].add(q)
                    self.P_open.add(q)
            self.remove_dominated()
            self.P_open -= self.G
            i += 1
            self.prune(i)
        P_candidates = self.get_candidates()
        return min(P_candidates, key=lambda plan: plan.cost)

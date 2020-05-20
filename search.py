# Standard libraries
import numpy as np
import collections


class Plan:
    def __init__(self, head_idx, path, cost, zonotope_state):
        self.head_idx = head_idx
        self.path = path
        self.cost = cost
        self.zonotope_state = zonotope_state

        self.n = 2 # replace this by taking the dimension from the arguments
        self.kmax = 100 # an argument with a default value?
        self.k = 0

        # coefficients that do not require the entire history
        self.a = np.eye(n)
        self.b = np.zeros(n)
        self.e = np.eye(n)

        # coefficients that require the entire history
        self.c = np.zeros((n,n,kmax))
        self.c[:,:,0] = np.eye(2)
        self.d = np.zeros((n,n,kmax))
        self.d[:,:,0] = np.zeros(2)
        self.p = np.zeros((n,n,kmax))
        self.q = np.zeros((n,n,kmax))

        # Kalman matrices
        self.C = None
        self.L = None
        self.Pbar = None
        self.Nu = None


    def add_point(self, env, point):
        '''
        - get C(k+1), R(k+1)
        - P = Pbar - L*C*Pbar
        - Pbar = A*P*A.T + Q
        - L = Eq (1.9)
        - update all Nu(n) if needed
        '''
        self.k += 1


    def update(self, A, B, K):
        '''
        - from k to k+1
        - assume L and C are already update by add_point
        '''
        M1 = A - B.dot(K)
        M2 = np.eye(self.n) - self.L.dot(self.C)

        self.a = M1.dot(self.a)
        self.b = M1.dot(self.b) - B.dot(K.dot(self.e))
        self.e = M2.dot(A.dot(self.e))

        self.c[:,:,self.k] = np.eye(2)
        self.p[:,:,self.k] = -M2
        self.q[:,:,self.k] = self.L

        # n = 0 ... self.k-1
        for n in range(self.k):
            self.c[:,:,n] = M1.dot(self.c[:,:,n]) - B.dot(K.dot(self.p[:,:,n]))
            self.d[:,:,n] = M1.dot(self.d[:,:,n]) - B.dot(K.dot(self.q[:,:,n]))
            self.p[:,:,n] = M2.dot(A.dot(self.p[:,:,n]))
            self.q[:,:,n] = M2.dot(A.dot(self.q[:,:,n]))


    def get_prob_unsafe(self, env):
        '''
        - called after add_point and update
        - now we have everything to compute all possible X(k+1)
          and intersect those with the associated environment
        '''
        pass


class Searcher:
    def __init__(self, graph, reachability_filter, pruning_coeff):
        self.graph = graph
        self.reachability_filter = reachability_filter
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

    def is_valid_goal(self):
        raise NotImplementedError

    def reached_goal(self):
        raise NotImplementedError

    def remove_dominated(self):
        raise NotImplementedError

    def prune(self, i):
        raise NotImplementedError

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

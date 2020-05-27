# Standard libraries
import numpy as np
import collections
from copy import deepcopy

# Custom libraries
from plan import Plan
from utils import GeoTools
from polytope import Polytope
from shapes import Point


class Searcher:
    def __init__(self, graph, pruning_coeff=0.5):
        self.graph = graph
        self.pruning_coeff = pruning_coeff
        self.x_init = None
        self.start_idx = None
        self.goal_region = None
        self.clear()
        self.prob_threshold = 1.0

    def set_source(self, start_idx=0):
        self.start_idx = start_idx
        self.x_init = self.graph.samples[start_idx, :]

    def initialize_open(self, Q, R, P0, kmax):
        self.clear()
        p_init = Plan(
            self.x_init, self.start_idx, self.graph.env, self.x_init.shape[0], R, kmax,
        )
        # Set variables in p_init
        p_init.set_motion(self.graph.planner.A, self.graph.planner.B, Q)
        p_init.set_gain(self.graph.planner.gain)
        p_init.set_init_est(P0)
        self.P_open.add(p_init)
        self.P[0] = self.P_open
        self.G = self.P_open

    def clear(self):
        self.P_open = set()
        self.P = collections.defaultdict(set)
        self.G = set()

    def set_goal(self, goal_region):
        """Input must be a Polytope object.
        """
        if not isinstance(goal_region, Polytope):
            print("Goal region must be a Polytope object.")
            raise
        self.goal_region = goal_region

    def is_valid_goal(self, config="worst", verbose=True):
        try:
            return GeoTools.is_valid_region(
                self.goal_region, self.graph.env, verbose, config=config
            )
        except AttributeError:
            print("Please set the goal region using .set_goal(goal_region).")

    def reached_goal(self):
        return len([p for p in self.G if self.in_goal(Point(p.head[0], p.head[1]))])

    def in_goal(self, point):
        if self.goal_region:
            return np.all(
                self.goal_region.A.dot(point.as_array()) <= self.goal_region.b
            )
        return False

    def remove_dominated(self):
        print("----- Calling remove_dominated -----")
        for p in self.P_open:
            for q_idx in self.P:
                if q_idx != p.head_idx:
                    continue
                else:
                    for q in self.P[q_idx]:
                        lower_cost = q.cost < p.cost
                        enclosed = q.Xk_full <= p.Xk_full
                        if lower_cost and enclosed:
                            self.P_open.remove(p)
                            self.P.remove(p)

    def prune(self, i):
        for p in self.P_open:
            if p.cost <= i * self.pruning_coeff:
                self.G.add(p)

    def collision(self, prob_collision):
        return self.prob_threshold <= prob_collision

    def get_candidates(self):
        P_candidates = set()
        for head_idx in self.P:
            if self.in_goal(
                Point(self.graph.samples[head_idx][0], self.graph.samples[head_idx][1])
            ):
                P_candidates.update(self.P[head_idx])

    def explore(self, prob_threshold=0.1, scaling_factors=[6, 5, 4, 3, 2, 1]):
        if self.x_init is None:
            print("Please set the source node using .set_source(x_init)")
            raise
        if len(self.P_open) == 0:
            print("Please initialize the searcher using .initialize_open(R, kmax).")
            raise
        try:
            assert self.is_valid_goal()
        except AssertionError:
            print("Invalid goal region.")
            raise
        self.prob_threshold = prob_threshold
        i = 0
        while self.P_open and not self.reached_goal():
            print("P_open = {}".format(self.P_open))
            for p in self.G:
                print(
                    "========== p = {} ==========".format(
                        self.graph.samples[p.head_idx]
                    )
                )
                for k, v in self.graph.edges[p.head_idx].items():
                    discard = False
                    neighbor_idx = k
                    to_neighbor_cost, to_neighbor_path = v
                    print("----- neighbor_idx = {} -----".format(neighbor_idx))
                    for sub_neighbor in to_neighbor_path[1:]:
                        print("== sub-neighbor = {} ==".format(sub_neighbor))
                        p.add_point(self.graph.env, sub_neighbor)
                        prob_collision = p.get_max_prob_collision(
                            self.graph.env, scaling_factors
                        )
                        print("prob_collision = {}".format(prob_collision))
                        if self.collision(prob_collision):
                            print("collided!")
                            discard = True
                            break
                    if discard:
                        print("discarded")
                        continue
                    print("adding to P and P_open")
                    p.update_info(neighbor_idx, to_neighbor_cost, to_neighbor_path)
                    q = p
                    print("P = {}".format(self.P))
                    print("P_open = {}".format(self.P_open))
                    self.P[neighbor_idx].add(p)
                    self.P_open.add(q)
                    print("P = {}".format(self.P))
                    print("P_open = {}".format(self.P_open))
            self.remove_dominated()
            self.P_open -= self.G
            i += 1
            self.prune(i)
        P_candidates = self.get_candidates()
        print("P_candidates = {}".format(P_candidates))
        if not P_candidates:
            print("Could not find a path.")
            return None
        return min(P_candidates, key=lambda plan: plan.cost)

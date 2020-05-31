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
        self.plan_number = 0
        p_init = Plan(
            self.x_init, self.start_idx, self.graph.env, self.x_init.shape[0], R, kmax,
        )
        # Set variables in p_init
        p_init.set_motion(self.graph.planner.A, self.graph.planner.B, Q)
        p_init.set_gain(self.graph.planner.gain)
        p_init.set_init_est(P0)
        self.P_open.add((p_init, self.plan_number))
        self.P[0] = set()
        self.P[0].add(deepcopy(p_init))
        self.G = self.P_open.copy()

    def clear(self):
        self.P_open = set()
        self.P = collections.defaultdict(set)
        self.G = set()

    def set_goal(self, goal_region):
        """Input must be a Rectangle object.
        """
        self.goal_region = goal_region.as_poly["original"]

    def is_valid_goal(self, config="worst", verbose=True):
        try:
            return GeoTools.is_valid_region(
                self.goal_region, self.graph.env, verbose, config=config
            )
        except AttributeError:
            print("Please set the goal region using .set_goal(goal_region).")

    def reached_goal(self, early_termination):
        if not early_termination:
            return False
        return len(
            [
                p
                for p, plan_number in self.G
                if self.in_goal(Point(p.head[0], p.head[1]))
            ]
        )

    def in_goal(self, point):
        if self.goal_region:
            return np.all(
                self.goal_region.A.dot(point.as_array()) <= self.goal_region.b
            )
        return False

    def remove_dominated(self):
        # print(
        #     "===================== [INFO] Calling remove_dominated ======================"
        # )
        # print("P_open before = {}".format(self.P_open))
        # print("P before = {}".format(self.P))
        for p, plan_number in self.P_open:
            if p.head_idx in self.P:
                for q in self.P[p.head_idx]:
                    lower_cost = q.cost < p.cost
                    enclosed = q.Xk_full <= p.Xk_full
                    if lower_cost and enclosed:
                        self.P_open.remove((p, plan_number))
                        self.P.remove(p)
        # print("P_open after = {}".format(self.P_open))
        # print("P after = {}".format(self.P))
        # print(
        #     "==========================================================================="
        # )

    def prune(self, i):
        # print(
        #     "=========================== [INFO] Calling prune ==========================="
        # )
        # print("G before = {}".format(self.G))
        self.G = set()
        for p, plan_number in self.P_open:
            # print("p.cost = {}".format(p.cost))
            # print("i * self.pruning_coeff = {}".format(i * self.pruning_coeff))
            if p.cost <= i * self.pruning_coeff:
                self.G.add((p, plan_number))
        # print("G after = {}".format(self.G))
        # print(
        #     "==========================================================================="
        # )

    def collision(self, prob_collision):
        return prob_collision >= self.prob_threshold

    def get_candidates(self):
        P_candidates = set()
        for head_idx in self.P:
            if self.in_goal(
                Point(self.graph.samples[head_idx][0], self.graph.samples[head_idx][1])
            ):
                P_candidates.update(self.P[head_idx])
        return P_candidates

    def explore(
        self,
        prob_threshold=0.1,
        scaling_factors=[6, 5, 4, 3, 2, 1],
        early_termination=True,
    ):
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
        while self.P_open and not self.reached_goal(early_termination):
            # print("P_open = {}".format(self.P_open))
            # print("G = {}".format(self.G))
            for p, plan_number in self.G:
                print(
                    "========== p = {} at index {} ==========".format(
                        self.graph.samples[p.head_idx], p.head_idx
                    )
                )
                print("NEIGHBORS: {}".format(self.graph.edges[p.head_idx].keys()))
                print("p.path_indices = {}".format(p.path_indices))
                for neighbor_idx, v in self.graph.edges[p.head_idx].items():
                    if neighbor_idx in set(p.path_indices):
                        continue
                    discard = False
                    to_neighbor_cost, to_neighbor_path = v
                    print("----- neighbor_idx = {} -----".format(neighbor_idx))
                    p_copy = deepcopy(p)
                    for sub_neighbor in to_neighbor_path[1:]:
                        # print("== sub-neighbor = {} ==".format(sub_neighbor))
                        p_copy.add_point(self.graph.env, sub_neighbor)
                        prob_collision = p_copy.get_max_prob_collision(
                            self.graph.env, scaling_factors
                        )
                        print("prob_collision = {}".format(prob_collision))
                        if self.collision(prob_collision):
                            print("prob_collision = {}".format(prob_collision))
                            print("collided!")
                            discard = True
                            break
                    if discard:
                        print("discarded")
                        continue
                    print("adding to P and P_open")
                    p_copy.update_info(neighbor_idx, to_neighbor_cost, to_neighbor_path)
                    self.plan_number += 1
                    self.P[neighbor_idx].add(p_copy)
                    self.P_open.add((p_copy, self.plan_number))
                    # print("P = {}".format(self.P))
                    # print("P_open = {}".format(self.P_open))
            self.remove_dominated()
            self.P_open -= self.G
            i += 1
            self.prune(i)
        P_candidates = self.get_candidates()
        print("P_candidates = {}".format(P_candidates))
        if not P_candidates:
            print("Could not find a path.")
            return None
        return P_candidates, min(P_candidates, key=lambda plan: plan.cost)
